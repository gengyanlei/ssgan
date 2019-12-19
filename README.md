# ssgan
Semi Supervised Semantic Segmentation Using Generative Adversarial Network ; Pytorch

Python：3.5 ；  Pytorch：0.40

复现论文：“Semi Supervised Semantic Segmentation Using Generative Adversarial Network”

由于论文未给出代码，并且此论文为“分割”SEMI-GAN，与分类有相似之处，但仍有巨大区别，在参考一些分类SEMI-GAN后，复现此半监督分割GAN论文代码。

若有问题，请及时指出。谢谢。

注意：测试时请添加model.eval() and with torch.no_grad(): 

分享 建筑物语义分割数据集： building-segmentation-dataset

https://github.com/gengyanlei/build_segmentation_dataset
