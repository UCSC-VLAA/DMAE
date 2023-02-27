

## Masked Autoencoders Enable Efficient Knowledge Distillers


This is a PyTorch implementation of the [DMAE paper](https://arxiv.org/abs/2208.12256).

<div align="center">
  <img src="dmae_teaser.png"/>
</div>


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Please refer to [MAE official codebase](https://github.com/facebookresearch/mae) for other enrironment requirements.



### Pre-Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To pre-train models in an 8-gpu machine, please first download the [ViT-Large model](https://drive.google.com/drive/folders/1tCdXhi_pWbRSgdUcmyOyP5mE0GMnpeC9?usp=sharing) as the teacher model, and then run:
```
bash pretrain.sh
```


### Finetuning
To fintune models in an 8-gpu machine, run:

```
bash finetune.sh
```


### Models

The checkpoints of our pre-trained and finetuned ViT-Base on ImageNet-1k can be downloaded as following:


|             |                                          Pretrained Model                                           | Epoch | 
| ----------- | :-------------------------------------------------------------------------------------------------: | :------: 
| ViT-Base   | [download link](https://drive.google.com/drive/folders/1tCdXhi_pWbRSgdUcmyOyP5mE0GMnpeC9?usp=sharing) |   100   | 



|             |                                          Finetuned Model                                           | Acc | 
| ----------- | :-------------------------------------------------------------------------------------------------: | :------: 
| ViT-Base   | [download link](https://drive.google.com/drive/folders/1tCdXhi_pWbRSgdUcmyOyP5mE0GMnpeC9?usp=sharing) |   84.0   | 





### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.




### Citation

```
@inproceedings{bai2022masked,
  title     = {Masked autoencoders enable efficient knowledge distillers},
  author    = {Bai, Yutong and Wang, Zeyu and Xiao, Junfei and Wei, Chen and Wang, Huiyu and Yuille, Alan and Zhou, Yuyin and Xie, Cihang},
  booktitle = {CVPR},
  year      = {2023}
}
```
