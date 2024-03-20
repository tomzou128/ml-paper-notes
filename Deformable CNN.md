## Deformable CNN

假设输入特征图高和宽分别是 h 和 w，第二部分的卷积核尺寸是 k_h 和 k_w。那么第一部分卷积层的卷积核数、通道数是 2×kh×kw，计算偏移的通道维与原本的卷积的输出通道数无关，这个2表示x轴和y轴两个方向上的偏移值，而且输出特征图的宽高和输入特征图的宽高一样，这样offset的维度就是 [batch_size, 2×kh×kw, h, w]，假如第二部分设置了group参数（group 定义与 `nn.conv2d` 里一样），那么第一部分的卷积核数量就是 2×kh×kw×group，相当于每一个group用一套offset。第二部分的deformable convolution可以看作是先基于第一部分生成的offset执行插值计算，然后再执行普通卷积操作的过程。

假设 `deform_groups = 2` 代表对每个卷积核生成两组 offset，每组 offset 形状 [2, k_h, k_w]，特征图中前半数通道使用第一组 offset 然后和卷积核计算，后半数通道使用第二组 offset。
