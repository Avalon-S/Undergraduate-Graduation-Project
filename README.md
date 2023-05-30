# Undergraduate-Graduation-Project
[Welcome to my homepage](https://avalon-s.github.io/)
## 介绍
基于非局部注意力机制的深度学习高分遥感语义分割方法研究</br>
本项目是本人的本科毕业设计，详细介绍可以查看本页面的[pdf文件](https://github.com/Avalon-S/Undergraduate-Graduation-Project/blob/main/%E5%9F%BA%E4%BA%8E%E9%9D%9E%E5%B1%80%E9%83%A8%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E9%AB%98%E5%88%86%E9%81%A5%E6%84%9F%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E6%96%B9%E6%B3%95%E7%A0%94%E7%A9%B6.pdf)
![model](https://github.com/Avalon-S/Undergraduate-Graduation-Project/blob/main/Undergraduate-Graduation-Project/figs/model.png)</br>
<div align=center>
Fig. 1.  模型整体架构
</div>
</br>

![Residual%20Attention%20Block](https://github.com/Avalon-S/Undergraduate-Graduation-Project/blob/main/Undergraduate-Graduation-Project/figs/Residual%20Attention%20Block.png)</br>
<div align=center>
Fig. 2.  残差加权注意力模块
</div>
</br>
 
![visualization_1](https://github.com/Avalon-S/Undergraduate-Graduation-Project/blob/main/Undergraduate-Graduation-Project/figs/visualization_1.png)</br>
<div align=center>
Fig. 3.  小图推理可视化
</div>
</br>

![visualization_2](https://github.com/Avalon-S/Undergraduate-Graduation-Project/blob/main/Undergraduate-Graduation-Project/figs/visualization_2.png)</br>
<div align=center>
Fig. 4.  大图推理可视化
</div>
</br>

## 数据预处理
此处可以参考[mmsegmentation的处理方法](https://mmsegmentation.readthedocs.io/zh_CN/latest/user_guides/2_dataset_prepare.html)
## 数据集文件夹组织结构
```none
├── dataset
│   ├── potsdam
│   │   ├── train
│   │   │   ├──images_512
│   │   │   ├──masks_512
│   │   ├── test
│   │   │   ├──images_512
│   │   │   ├──masks_512
│   ├── vaihingen (the same with postdam)
```
## 训练
Potsdam：
```
python Undergraduate-Graduation-Project/train_supervision.py -c Undergraduate-Graduation-Project/config/potsdam/unetformer.py
```
Vaihingen:
```
python Undergraduate-Graduation-Project/train_supervision.py -c Undergraduate-Graduation-Project/config/vaihingen/unetformer.py
```
## 测试
Potsdam：
```
python Undergraduate-Graduation-Project/potsdam_test.py -c Undergraduate-Graduation-Project/config/potsdam/unetformer.py -o fig_results/potsdam/sk --rgb -t 'lr'
```
Vaihingen:
```
python Undergraduate-Graduation-Project/vaihingen_test.py -c Undergraduate-Graduation-Project/config/vaihingen/unetformer.py -o fig_results/vaihingen/sk --rgb -t 'lr'
```
## 大图可视化
Potsdam：
```
python Undergraduate-Graduation-Project/inference_huge_image.py -i test_images/Potsdam -c Undergraduate-Graduation-Project/config/potsdam/unetformer.py -o fig_results/potsdam/SK_huge -t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```
Vaihingen:
```
python Undergraduate-Graduation-Project/inference_huge_image.py -i test_images/Vaihingen -c Undergraduate-Graduation-Project/config/vaihingen/unetformer.py -o fig_results/vaihingen/SK_huge -t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```
## 感谢
本项目参考了Wang的[GeoSeg](https://github.com/WangLibo1995/GeoSeg)项目，此处向他表示真心的感谢。
