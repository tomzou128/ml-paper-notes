## DDP

https://pytorch.org/tutorials/beginner/ddp_series_intro.html



### Why you should prefer DDP over `DataParallel` (DP)

[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) is an older approach to data parallelism. DP is trivially simple (with just one extra line of code) but it is much less performant. DDP improves upon the architecture in a few ways:

| `DataParallel`                                               | `DistributedDataParallel`                                  |
| ------------------------------------------------------------ | ---------------------------------------------------------- |
| More overhead; model is replicated and destroyed at each forward pass | Model is replicated only once                              |
| Only supports single-node parallelism                        | Supports scaling to multiple machines                      |
| Slower; uses multithreading on a single process and runs into Global Interpreter Lock (GIL) contention | Faster (no GIL contention) because it uses multiprocessing |





## Single Node Single GPU

### 执行

```
python single_gpu.py 50 10
```



**数据集类**

生成随机数据，不受 GPU 数量影响。

`datautils.py`

```py
import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
```



### 模型训练

**导包**

```py
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset
```

**创建主方法**

创建主方法、数据集、模型和优化器，调用模型训练类进行训练。

`single_gpu.py`

```py
# 创建数据集、模型和优化器对象，与分布式无关
def load_train_objs():
    train_set = MyTrainDataset(4096)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

# 创建数据加载器对象，与分布式有关
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

# 主方法，与分布式有关
def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
	
    # 指定使用 GPU 0
    device = 0  # shorthand for cuda:0
    main(device, args.total_epochs, args.save_every, args.batch_size)
```

**模型训练类**

管理模型前向、梯度更新和模型保存。

`single_gpu.py`

```py
class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
	
    # 训练一个批量
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
	
    # 训练一个 epoch
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
	
    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
	
    # train -> run_epoch -> run_batch
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
```





## Single Node Multi GPU

### Run

本程序自动获取本节点全部 GPU，所以命令和之前一样。

```
python multigpu.py 50 10
```

**注意事项**

- IO 方法应在主进程运行，如打印进度和保存模型。可以用 `rank == 0` 判断当前是否在主进程。



### Imports

额外导入以下内容：

`multi_gpu.py`

```py
# Python 多进程的包装类
import torch.multiprocessing as mp
# 用于数据并行
from torch.utils.data.distributed import DistributedSampler
# 用于模型分布
from torch.nn.parallel import DistributedDataParallel as DDP
# 用于流程初始化
from torch.distributed import init_process_group, destroy_process_group
# 用于读取环境变量
import os
```



### Constructing the process group

流程初始化。首先读取 GPU 数量并创建对应数量的进程，传入总 GPU 数量。进程的 GPU 会自动传给进程方法。

- `world_size` is the number of processes across the training job. For GPU training, this corresponds to the number of GPUs in use, and each process works on a dedicated GPU.

```py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
```

前两个参数位置接收分布训练的参数，并传给流程初始化方法。

- Include new arguments `rank` (replacing `device`) and `world_size`.
- `rank` is auto-allocated by DDP when calling [mp.spawn](https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses).

```py
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()
```

创建流程初始化方法。

- The process group can be initialized by TCP (default) or from a shared file-system. Read more on [process group initialization](https://pytorch.org/docs/stable/distributed.html#tcp-initialization)
- [init_process_group](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group) initializes the distributed process group.
- Read more about [choosing a DDP backend](https://pytorch.org/docs/stable/distributed.html#which-backend-to-use)
- [set_device](https://pytorch.org/docs/stable/generated/torch.cuda.set_device.html?highlight=set_device#torch.cuda.set_device) sets the default GPU for each process. This is important to prevent hangs or excessive memory utilization on GPU:0

```py
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # The distributed process group contains all the processes that can communicate and synchronize with each other.
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
```



### Distributing input data

数据分发。**传入的批量依旧是单个 GPU 上的批量**，方法自动计算多个 GPU 时的情况。此处仅改动数据加载器，使其用 `DistributedSampler()`。并通过它设置数据是否乱序。

- [DistributedSampler](https://pytorch.org/docs/stable/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler) chunks the input data across all distributed processes.
- **Each process will receive an input batch of 32 samples**; the effective batch size is `32 * nprocs`, or 128 when using 4 GPUs.

```py
def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset, shuffle=True)
    )
```

并且在每个 epoch 前给 sampler 传入当前 epoch 数，才能正常使用数据乱序。

- Calling the `set_epoch()` method on the `DistributedSampler` at the beginning of each epoch is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be used in each epoch.

```py
def _run_epoch(self, epoch):
    b_sz = len(next(iter(self.train_data))[0])
    print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
    self.train_data.sampler.set_epoch(epoch)   # 传入 epoch
    for source, targets in self.train_data:
        source = source.to(self.gpu_id)
        targets = targets.to(self.gpu_id)
        self._run_batch(source, targets)
```



### Constructing the DDP model

创建分布式的模型。该方法不影响模型权重，所以**在这一步前或后创建模型优化器都没有影响**。

```py
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

        # Wrap using DDP
        self.model = DDP(model, device_ids=[gpu_id])
```



### Saving model checkpoints

最后在主进程保存模型，要保存 `model.model.state_dict()` 而不是 `model.state_dict()`。

```py
def train(self, max_epochs: int):
    for epoch in range(max_epochs):
        self._run_epoch(epoch)
        # Only save on GPU 0
        if self.gpu_id == 0 and epoch % self.save_every == 0:
            self._save_checkpoint(epoch)

def _save_checkpoint(self, epoch):
    # Get real model from DDP wrapped model
    ckp = self.model.module.state_dict()
    PATH = "checkpoint.pt"
    torch.save(ckp, PATH)
    print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
```





## Fault-Tolerant Distributed Training with `torchrun`

In distributed training, a single process failure can disrupt the entire training job. Since the susceptibility for failure can be higher here, making your training script robust is particularly important here. You might also prefer your training job to be *elastic*, for example, compute resources can join and leave dynamically over the course of the job.

PyTorch offers a utility called `torchrun` that provides fault-tolerance and elastic training. When a failure occurs, `torchrun` logs the errors and attempts to automatically restart all the processes from the last saved “snapshot” of the training job.

The snapshot saves more than just the model state; it can include details about the number of epochs run, optimizer states or any other stateful attribute of the training job necessary for its continuity.

### Why use `torchrun`

`torchrun` handles the minutiae of distributed training so that you don’t need to. For instance,

- You don’t need to set environment variables or explicitly pass the `rank` and `world_size`; `torchrun` assigns this along with several other [environment variables](https://pytorch.org/docs/stable/elastic/run.html#environment-variables).
- No need to call `mp.spawn` in your script; you only need a generic `main()` entry point, and launch the script with `torchrun`. This way the same script can be run in non-distributed as well as single-node and multinode setups.
- Gracefully restarting training from the last saved training snapshot.



### Graceful restarts

For graceful restarts, you should structure your train script like:

```py
def main():
  load_snapshot(snapshot_path)
  initialize()
  train()

def train():
  for batch in iter(dataset):
    train_step(batch)

    if should_checkpoint:
      save_snapshot(snapshot_path)
```

If a failure occurs, `torchrun` will terminate all the processes and restart them. Each process entry point first loads and initializes the last saved snapshot, and continues training from there. So at any failure, you only lose the training progress from the last saved snapshot.

In elastic training, whenever there are any membership changes (adding or removing nodes), `torchrun` will terminate and spawn processes on available devices. Having this structure ensures your training job can continue without manual intervention.



### Run

```py
torchrun --standalone --nproc_per_node=2 multigpu_torchrun.py 50 10
```



### Modify DDP Setup

`torchrun` assigns `RANK` and `WORLD_SIZE` automatically, among [other envvariables](https://pytorch.org/docs/stable/elastic/run.html#environment-variables). 

```py
def ddp_setup():
    init_process_group(backend="gloo")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


```

Simply call your entry point function as you would for a non-multiprocessing script; `torchrun` automatically spawns the processes.

```py
def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    # No need to spawn manually
    main(args.save_every, args.total_epochs, args.batch_size)
```





### Modify Trainer Constructor

Use torchrun-provided environment variables and load the snapshot when start.

```py
class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
    ) -> None:
        # Change to read from env
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        # Load from snapshot if exsits
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])
```



### Saving and loading snapshots

Regularly storing the model and current epoch in snapshots allows our training job to seamlessly resume after an interruption.

`multigpu_torchrun.Trainer`

```py
def _load_snapshot(self, snapshot_path):
    loc = f"cuda:{self.gpu_id}"
    snapshot = torch.load(snapshot_path, map_location=loc)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

def _save_snapshot(self, epoch):
    snapshot = {
        "MODEL_STATE": self.model.module.state_dict(),
        "EPOCHS_RUN": epoch,
    }
    torch.save(snapshot, self.snapshot_path)
    print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
    
def train(self, max_epochs: int):
    for epoch in range(self.epochs_run, max_epochs):
        self._run_epoch(epoch)
        # Save snapshot often
        if self.gpu_id == 0 and epoch % self.save_every == 0:
            self._save_snapshot(epoch)
```





## Multi Node Training

https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html



## minGPT Training

https://pytorch.org/tutorials/intermediate/ddp_series_minGPT.html