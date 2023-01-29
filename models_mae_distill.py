from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block, Mlp

from util.pos_embed import get_2d_sincos_pos_embed


import torch.nn.functional as F



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, mixup_disentangled_target=False,
                 embedding_distillation_func=None, aligned_blks_indices=None,
                 distillation_disentangled_target=None, student_reconstruction_target='original_image',
                 aligned_feature_projection_mode=None, aligned_feature_projection_dim=None, dropout=0.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.dropout = dropout
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if mixup_disentangled_target:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans * 3, bias=True)
        elif distillation_disentangled_target is not None:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans *
                                          distillation_disentangled_target, bias=True)
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.aligned_blks_indices = aligned_blks_indices

        if self.aligned_blks_indices is not None:
            assert embedding_distillation_func is not None
            distillation_loss_dict = dict(L1=nn.L1Loss(), L2=nn.MSELoss())
            self.distillation_criterion = distillation_loss_dict[embedding_distillation_func]

        self.initialize_weights()

        self.student_reconstruction_target = student_reconstruction_target

        if aligned_feature_projection_mode is not None:
            assert aligned_feature_projection_dim is not None
            assert aligned_feature_projection_dim[0] == embed_dim
            if aligned_feature_projection_mode == 'fc-1layer':
                student_feature_dim, teacher_feature_dim = aligned_feature_projection_dim
                self.aligned_feature_projection_heads = nn.ModuleList([
                    nn.Linear(student_feature_dim, teacher_feature_dim)
                    for i in range(len(self.aligned_blks_indices))]
                )
            elif aligned_feature_projection_mode == 'mlp-1layer':
                student_feature_dim, teacher_feature_dim = aligned_feature_projection_dim
                self.aligned_feature_projection_heads = nn.ModuleList([
                    Mlp(in_features=student_feature_dim, hidden_features=teacher_feature_dim, out_features= teacher_feature_dim, act_layer=nn.GELU, drop=self.dropout)
                    for i in range(len(self.aligned_blks_indices))]
                )
            elif aligned_feature_projection_mode == 'mlp-2layer':
                student_feature_dim, teacher_feature_dim = aligned_feature_projection_dim
                self.aligned_feature_projection_heads = nn.ModuleList([
                    nn.Sequential(*[
                    Mlp(in_features=_feature_dim, hidden_features=teacher_feature_dim, out_features= teacher_feature_dim, act_layer=nn.GELU, drop=0.0),
                    Mlp(in_features=teacher_feature_dim, hidden_features=teacher_feature_dim, out_features= teacher_feature_dim, act_layer=nn.GELU, drop=0.0)])
                    for i in range(len(self.aligned_blks_indices))]
                )
        else:
            self.aligned_feature_projection_heads = None

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_customized(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_customized(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore, ids_keep = self.random_masking_customized(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outs = []
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == len(self.blocks) - 1 and self.aligned_blks_indices is None:
                x = self.norm(x)
            if self.aligned_blks_indices is not None:
                if i in self.aligned_blks_indices:
                    x = x.clone().to(torch.float32)
                    outs.append(x)
                if i == len(self.blocks) - 1:
                    x = self.norm(x)
                    x = x.clone().to(torch.float32)
                    outs.append(x)

        if self.aligned_blks_indices is None:
            return x, mask, ids_restore, ids_keep
        else:
            return outs, mask, ids_restore, ids_keep

    def forward_encoder_student(self, x, ids_keep):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        N, L, D = x.shape
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore, ids_keep = self.random_masking_customized(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outs = []
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i == len(self.blocks) - 1 and self.aligned_blks_indices is None:
                x = self.norm(x)

            if self.aligned_blks_indices is not None:
                if i in self.aligned_blks_indices:
                    outs.append(x)
                if i == len(self.blocks) - 1:
                    x = self.norm(x)
                    outs.append(x)

        if self.aligned_blks_indices is None:
            return x
        else:
            return outs

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_disentangle_mixup(self, imgs, pred, mask, imgs_mixuped):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target1 = self.patchify(imgs)
        target2 = self.patchify(imgs.flip(0))
        target3 = self.patchify(imgs_mixuped)
        target_list = [target1, target2, target3]
        if self.norm_pix_loss:
            for i, target in enumerate(target_list):
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target_list[i] = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - torch.cat(target_list, dim=2)) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_distillation_loss_embedding(self, features_teacher, features_student):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        assert isinstance(features_teacher, list) and isinstance(features_student, list)
        assert len(features_teacher) == len(features_student)
        loss_distillation_embedding = dict()
        if self.aligned_feature_projection_heads is not None:
            for feature_teacher, feature_student, blk_idx, projection_head in zip(
                    features_teacher, features_student,
                    self.aligned_blks_indices, self.aligned_feature_projection_heads):
                loss_distillation_embedding[f'align_block{blk_idx}'] = \
                    self.distillation_criterion(F.normalize(feature_teacher.detach(), dim=-1), F.normalize(projection_head(feature_student), dim=-1))
        else:
            for feature_teacher, feature_student, blk_idx in zip(features_teacher, features_student,
                                                                 self.aligned_blks_indices):
                loss_distillation_embedding[f'align_block{blk_idx}'] = \
                    self.distillation_criterion(feature_teacher.detach(), feature_student)

        return loss_distillation_embedding

    def forward_loss_student(self, teacher_pred, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        loss = (pred - teacher_pred) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_student_diff(self, imgs, teacher_pred, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        diff = target - teacher_pred

        loss = (pred - diff) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_student_weighted_sum(self, imgs, teacher_pred, pred, mask, weights=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        if weights is None:
            original_img_weight = 0.5
            teacher_pred_weight = 0.5
        else:
            original_img_weight, teacher_pred_weight = weights

        weighted_sum = original_img_weight * target + teacher_pred_weight * teacher_pred

        loss = (pred - weighted_sum) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_student_disentangled(self, imgs, teacher_prediction, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target1 = self.patchify(imgs)
        target2 = teacher_prediction

        if self.norm_pix_loss:
            mean = target1.mean(dim=-1, keepdim=True)
            var = target1.var(dim=-1, keepdim=True)
            target1 = (target1 - mean) / (var + 1.e-6) ** .5

        target_list = [target1, target2]
        loss = (pred - torch.cat(target_list, dim=2)) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, ids_keep, ids_restore, mask, teacher_prediction, target_sum_weights=None,
                latents_teacher=None):

        assert latents_teacher is not None
        latents = self.forward_encoder_student(imgs, ids_keep)
        pred = self.forward_decoder(latents[-1], ids_restore)  # [N, L, p*p*3]


        loss_distillation_embedding = self.forward_distillation_loss_embedding(latents_teacher[:-1],
                                                                                    latents[:-1])
        if self.student_reconstruction_target == 'original_img':
            loss = self.forward_loss(imgs, pred, mask)
        else:
            raise NotImplementedError
        return loss, loss_distillation_embedding, pred, mask


def mae_vit_small_patch16_dec512d2b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d2b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
