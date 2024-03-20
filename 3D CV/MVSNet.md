# MVSNet



引言



方法



Differentiable 





## 数据处理

数据集







```
MVSDataset():
初始化
- 数据集位置
- 数据集 scan 文件夹列表文件位置
- nviews: 
- ndepths:
- interval_scal
```





## 模型

建构式

```py
def __init__(self, refine=True):
    super(MVSNet, self).__init__()
    self.refine = refine

    self.feature = FeatureNet()   # feature extraction network, 2D CNN
    self.cost_regularization = CostRegNet()
    if self.refine:
        self.refine_network = RefineNet()
```

