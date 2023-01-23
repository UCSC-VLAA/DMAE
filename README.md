

## Masked Autoencoders Enable Efficient Knowledge Distillers


This is a PyTorch implementation of the [DMAE paper](https://arxiv.org/abs/2208.12256):
```
@article{bai2022masked,
  title={Masked autoencoders enable efficient knowledge distillers},
  author={Bai, Yutong and Wang, Zeyu and Xiao, Junfei and Wei, Chen and Wang, Huiyu and Yuille, Alan and Zhou, Yuyin and Xie, Cihang},
  journal={arXiv preprint arXiv:2208.12256},
  year={2022}
}
```


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). Refer to MAE official codebase for other enrironment.



### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training in an 8-gpu machine,

first, download the [ViT-Large model]((https://drive.google.com/drive/folders/1tCdXhi_pWbRSgdUcmyOyP5mE0GMnpeC9?usp=sharing)) as teacher model, and run:
```
bash pretrain.sh
```


### Supervised Finetuning
To do fintuning in an 8-gpu machine, run:

```
bash finetune.sh
```


### Models

Our pre-trained ResNet-50 model and finetuned checkpoints on object detection can be downloaded as following:


|             |                                          Pretrained Model                                           | Epoch | 
| ----------- | :-------------------------------------------------------------------------------------------------: | :------: 
| ViT-Base   | [download link](https://drive.google.com/drive/folders/1tCdXhi_pWbRSgdUcmyOyP5mE0GMnpeC9?usp=sharing) |   100   | 



|             |                                          Finetuned Model                                           | Acc | 
| ----------- | :-------------------------------------------------------------------------------------------------: | :------: 
| ViT-Base   | [download link](https://drive.google.com/drive/folders/1tCdXhi_pWbRSgdUcmyOyP5mE0GMnpeC9?usp=sharing) |   84.0   | 




### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
