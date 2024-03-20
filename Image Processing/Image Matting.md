## Image Matting 抠图

### 数据集

参考：https://blog.csdn.net/qq_41731861/article/details/121927295

#### PPM-100

- 人像抠图数据集，100 个样本，尺寸和背景都很多元化
- 包含原图和 alpha matte 图

由于数据量小，通常作为测试集使用。MODNet 中所使用的数据集。
链接：https://github.com/ZHKKKe/PPM



#### 爱分割 AISegment

- 人像抠图数据集，34k+ 个样本，都为 $600 \times 800$ 分辨率的半身人像图
- 包含原图和 FG 图，FG 图可用于计算 alpha matte 图

链接：https://github.com/aisegmentcn/matting_human_datasets



#### Adobe Image Matting Dataset (AIM)

- 人像和物体抠图数据集
- 包含原图和 alpha matte 图

非公开，联系作者 bprice@adobe.com



#### Distinctions-646 (D646)

- 人像和物体抠图数据集，646 个样本
- 包含 FG GT

非公开，联系作者：coachqiao2018@gmail.com

https://github.com/yuhaoliu7456/CVPR2020-HAttMatting



#### Alpha Matting

一些布偶的图像及其 groundtruth，基本不含人像。

链接：http://www.alphamatting.com/datasets.php



#### AIM-500

- 人像和物体抠图数据集，500 个样本，其中 100 个为人像
- 包含原图、alpha matte 图、trimap 和 unified semantic representation

链接：https://github.com/JizhiziLi/AIM



#### DUTS

DUTS 数据集包含 10553 个训练图像和 5019 个测试图像，所有的训练图像均从 ImageNet DET 训练/验证集中收集，而测试图像则从 ImageNet DET 测试集和 SUN数据集中收集。训练和测试集都包含非常重要的场景用于显著性检测。

链接：https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset



#### RefMatte

- 人像和物体抠图数据集

大型数据集，链接：https://github.com/jizhiziLi/rim



#### P3M-10K

**P3M-10k Dataset**: To further explore the effect of PPT setting, we establish the first large-scale privacy-preserving portrait matting benchmark named P3M-10k. It contains 10,000 annonymized high-resolution portrait images by face obfuscation along with high-quality ground truth alpha mattes. Specifically, we carefully collect, filter, and annotate about **10,000** high-resolution images from the Internet with free use license. There are **9,421** images in the training set and **500** images in the test set, denoted as **P3M-500-P**. In addition, we also collect and annotate another **500** public celebrity images from the Internet without face obfuscation, to evaluate the performance of matting models under the PPT setting on normal portrait images, denoted as **P3M-500-NP**. We show some examples as below, where (a) is from the training set, (b) is from **P3M-500-P**, and (c) is from **P3M-500-NP**.

https://github.com/JizhiziLi/P3M



#### HIM2K

human instance matting dataset.

https://github.com/nowsyn/InstMatt/tree/main



### 背景数据集

用于与任意前景物体制作合成图片。

#### BG-20K

- 高清背景数据集，2 万张图片

链接：https://github.com/JizhiziLi/GFM