# Neural Recon

NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video



## 代码实现

### 环境配置

```
cuda=11.3
```



缺少 `libcudnn_cnn_infer.so.8.`

```py
sudo apt install nvidia-cudnn
```

然后添加到 PATH

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```





### 准备 GT

**训练集**

```
python tools/tsdf_fusion/generate_gt.py --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9
```

**测试集**

```
python tools/tsdf_fusion/generate_gt.py --test --data_path PATH_TO_SCANNET --save_name all_tsdf_9 --window_size 9
```



**脚本参数**

- `--n_gpu`：使用的 GPU 个数
- `n_proc`：每个 GPU 上的进程数，默认 16。如 1 个 GPU + 16 进程代表每个进程处理 94~95 个场景



**读取数据**

一个场景一个场景处理。修改为适合读取 key frame 版本的数据集。

```py
# 图片数
n_imgs = len(os.listdir(os.path.join(args.data_path, scene, 'color')))
# 读取内参矩阵
intrinsic_dir = os.path.join(args.data_path, scene, 'intrinsics_depth.txt')
cam_intr = np.loadtxt(intrinsic_dir)[:3, :3]
# 获得场景里所有图片的名称
id_list = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(args.data_path, scene, 'color'))]
# 生成数据集
dataset = ScanNetDataset(n_imgs, scene, args.data_path, args.max_depth, id_list)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, collate_fn=collate_fn,
	batch_sampler=None, num_workers=args.loader_num_workers)

# 将深度图和外参矩阵存储起来
for id, cam_pose, depth_im, _ in dataloader:
    if cam_pose[0][0] == np.inf or cam_pose[0][0] == -np.inf or cam_pose[0][0] == np.nan:
        continue
    depth_all.update({id: depth_im})
    cam_pose_all.update({id: cam_pose})
    # color_all.update({id: color_image})

# 计算 GT TSDF
save_tsdf_full(args, scene, cam_intr, depth_all, cam_pose_all, color_all, save_mesh=False)
save_fragment_pkl(args, scene, cam_intr, depth_all, cam_pose_all)
```



**`save_tsdf_full()`**

计算一个场景的 TSDF，若场景中的图片大于200张，会使用 `linspace()` 均匀采样 200 张。首先计算场景中 volume 的长宽高。 

```py
for id in image_id:
    depth_im = depth_list[id]
    cam_pose = cam_pose_list[id]

    # Compute camera view frustum and extend convex hull
    view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
```

由于 NeuralRecon 是 coarse-to-fine 的结构，共有 3 个 stage。所以生成每个 stage 的所有体素块。体素大小分别是 0.04, 0.08, 0.16m。

```py
tsdf_vol_list = []
for l in range(args.num_layers):
    tsdf_vol_list.append(TSDFVolume(vol_bnds, voxel_size=args.voxel_size * 2 ** l, margin=args.margin))
```

然后更新 TSDF，每个 stage 单独计算。

```py
for id in depth_list.keys():
    depth_im = depth_list[id]
    cam_pose = cam_pose_list[id]
    if len(color_list) == 0:
        color_image = None
    else:
        color_image = color_list[id]

    # Integrate observation into voxel volume (assume color aligned with depth)
    for l in range(args.num_layers):
        tsdf_vol_list[l].integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
```

**存储该场景的体素原点的世界坐标和体素大小** (0.04)

```py
tsdf_info = {
    'vol_origin': tsdf_vol_list[0]._vol_origin,
    'voxel_size': tsdf_vol_list[0]._voxel_size,
}
tsdf_path = os.path.join(args.save_path, scene_path)
if not os.path.exists(tsdf_path):
    os.makedirs(tsdf_path)

with open(os.path.join(args.save_path, scene_path, 'tsdf_info.pkl'), 'wb') as f:
    pickle.dump(tsdf_info, f)
```

**存储该场景所有 stage 的 TSDF 标签**。

```py
for l in range(args.num_layers):
    tsdf_vol, color_vol, weight_vol = tsdf_vol_list[l].get_volume()
    np.savez_compressed(os.path.join(args.save_path, scene_path, 'full_tsdf_layer{}'.format(str(l))), tsdf_vol)
```

save mesh...



**生成配对信息**

**`save_fragment_pkl()`**

计算图片配对信息，使用外参矩阵计算图片间的相机角度，要大于一定程度才会配对。**存储该场景的图片配对信息、体素原点的世界坐标和体素大小**。

```py
for i, bnds in all_bnds.items():
    if not os.path.exists(os.path.join(args.save_path, scene, 'fragments', str(i))):
        os.makedirs(os.path.join(args.save_path, scene, 'fragments', str(i)))
    fragments.append({
        'scene': scene,
        'fragment_id': i,
        'image_ids': all_ids[i],
        'vol_origin': tsdf_info['vol_origin'],
        'voxel_size': tsdf_info['voxel_size'],
    })

with open(os.path.join(args.save_path, scene, 'fragments.pkl'), 'wb') as f:
    pickle.dump(fragments, f)
```



**`generate_pkl()`**

分别合并训练集、验证集和测试集所有的 `fragments.pkl`。每一行作为一个样本。存储为 `fragments_train`, `fragments_val` 和 `fragments_test`。

训练时读取 `fragments_train` 到 `metas`，读取每个场景每个 stage 的 TSDF 标签 `full_tsdf_layer0/1/2.npz`。



### 创建模型



### 读取数据

**创建数据集**

创建数据集时 读取所有配对信息，每行配对信息作为一个样本，和 NeuralRecon 的处理一样。

```py
def build_list(self):
    if self.scene is None:
        # load data for all scenes in the train/val/test split
        path = os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode))
    else:
        # load data for a specific scene
        path = os.path.join(self.datapath, self.tsdf_file, self.scene, 'fragments.pkl')
    with open(path, 'rb') as f:
        metas = pickle.load(f)
    return metas
```

读取单个样本时读取以下数据：

```py
items = {
    'imgs': imgs,                       # 9 images
    'depth': depth,                     # 9 depth map
    'intrinsics': intrinsics,           # (9, 3, 3)
    'extrinsics': extrinsics,           # (9, 4, 4)
    'tsdf_list_full': tsdf_list,        # 3 TSDF for 3 stages, no. of vox in each scene aren't same
    'vol_origin': meta['vol_origin'],   # (3,), voxel origin in world coordinate
    'scene': meta['scene'],             # string, scene/folder name
    'fragment': meta['scene'] + '_' + str(meta['fragment_id']),     # scene name + fragment name
    'epoch': [self.epoch],              # for random, value from: TrainImgLoader.dataset.epoch = epoch_idx
}
```

之后进行数据增强和变换，

