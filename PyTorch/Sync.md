### SyncBatchNorm

The mean and standard-deviation are calculated per-dimension over all mini-batches of the same process groups. $\gamma$ and $\beta$ are learnable parameter vectors of size C (where C is the input size). By default, the elements of $\gamma$ are sampled from $\mathcal{U}(0, 1)$ and the elements of $\beta$ are set to 0. The standard-deviation is calculated via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

Currently [`SyncBatchNorm`](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm) only supports `DistributedDataParallel` (DDP) with single GPU per process. Use [`torch.nn.SyncBatchNorm.convert_sync_batchnorm()`](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm) to convert `BatchNorm*D` layer to [`SyncBatchNorm`](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm) before wrapping Network with DDP.



`*CLASSMETHOD* convert_sync_batchnorm(*module*, *process_group=None*)`

Helper function to convert all `BatchNorm*D` layers in the model to [`torch.nn.SyncBatchNorm`](https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm) layers.