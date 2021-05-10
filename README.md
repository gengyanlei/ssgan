# ssgan
Semi Supervised Semantic Segmentation Using Generative Adversarial Network ; Pytorch

### Environment
```
    python：3.5 
    Pytorch：0.40
```

### Note
```
    由于论文未给出代码，并且此论文为“分割”SEMI-GAN，与分类有相似之处，但仍有巨大区别，
    在参考一些分类SEMI-GAN后，复现此半监督分割GAN论文代码。
    若有问题，请及时指出，谢谢。
    
    注意：测试时请添加model.eval() and with torch.no_grad(): 
```

### Refer
+ [Semi Supervised Semantic Segmentation Using Generative Adversarial Network](https://arxiv.org/abs/1703.09695)

### Other
+ [segmentation_pytorch](https://github.com/gengyanlei/segmentation_pytorch)
+ [building-segmentation-dataset](https://github.com/gengyanlei/build_segmentation_dataset)
+ [fire-smoke-detect-dataset](https://github.com/gengyanlei/fire-detect-yolov4)
