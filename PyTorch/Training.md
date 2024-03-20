## PyTorch



### `model.eval()` 和 `with torch.no_grad()` 的区别

#### `model.eval()`

使用 PyTorch 进行验证或测试时，会使用 `model.eval()` 将网络切换到测试模式，在该模式下：

- 将 dropout 层和标准化层设置为测试模式，