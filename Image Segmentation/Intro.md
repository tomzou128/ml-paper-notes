## Image/Semantic Segmentation 语义分割

**常见图像任务**

- Image Classification：图像分类，将一张图片判定为一个类别。
- Object Detection：物体检测，检测出一张图片中存在的多个物体，用方框的坐标表示物体的区域，并判定每个物体的类别。
- Semantic Segmentation：语义分割，即本任务，检测出一张图片中存在的多个物体，描述每个物体对应的具体像素，并判定每个物体的类别。
- Instance Segmentation：实例分割，在语义分割的基础上，将图片中多个同类别的物体分开用单独的实例表示。



常见数据集格式



### 常见语义分割评价指标

- Pixel Accuracy（Global Acc）：总共预测正确的像素个数 / 目标的总像素个数
- mena Accuracy：每个目标的 Acc，然后目标求均值
- mean IoU：每个目标的IoU再求平均