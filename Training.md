## PyTorch



### `model.eval()` 和 `with torch.no_grad()` 的区别

#### `model.eval()`

使用 PyTorch 进行验证或测试时，会使用 `model.eval()` 将网络切换到测试模式，在该模式下：

- 将 dropout 层和标准化层设置为测试模式，





model.eval()和with torch.no_grad()的区别
在PyTorch中进行validation时，会使用model.eval()切换到测试模式，在该模式下，

- 主要用于通知dropout层和batchnorm层在train和val模式间切换
  - 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新。
  - 在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
- 该模式不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反传（backprobagation）
- 而with torch.no_grad()则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。
  使用场景

如果不在意显存大小和计算时间的话，仅仅使用model.eval()已足够得到正确的validation的结果；而with torch.no_grad()则是更进一步加速和节省gpu空间（因为不用计算和存储gradient），从而可以更快计算，也可以跑更大的batch来测试。

参考
https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/38
https://ryankresse.com/batchnorm-dropout-and-eval-in-pytorch/