# 半监督迁移学习的自适应一致性正则化


##关于代码
该代码是在 CentOS 6.3 环境（Python 3.6、PyTorch 1.1、CUDA 9.0）和 Tesla V100 GPU 上开发的。
请查看[train.py](ssl_lib/trainer/train.py)中代码的核心训练部分； 检查 [regularizer.py](ssl_lib/consistency/regularizer.py) 中 ARC 和 AKC 的实现。
对于[parser.py](parser.py)中的超参数，`lambda_kd`代表AKC的正则化权重因子，`lambda_mmd`代表ARC的正则化权重因子；
`kd_threshold` 和 `mmd_threshold` 是 AKC 和 ARC 的阈值；
`mmd_feat_table_l` 和 `mmd_feat_table_u` 表示标记数据和未标记数据的 ARC 缓冲区大小。

## 怎样运行代码

### 运行 CUB-200-2011

1) 在您的数据文件夹中下载 [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) , e.g. `./data/`.
2) 在 [Imagenet](http://image-net.org/download-images) 上预训练 ResNet-50 模型或从 [pytorch.models](https://download.pytorch.org/) 下载 Imagenet 预训练模型 模型/resnet50-19c8e357.pth）在您的检查点文件夹中, e.g. `./ckpt/`. 然后将预训练的检查点重命名为'resnet_50_1.pth'。 
3) 在[parser.py](parser.py)中，`lambda_kd`代表AKC的正则化权重因子，`lambda_mmd`代表ARC的正则化权重因子。
4)在 CUB-200-2011 数据集上运行 [main.py](main.py)。
 
e.g:使用 ARC 训练模型：
```
pretrain_path="ckpt" # pretrained model path
data_root="data" # data folder
dataset=cub200
num_labels=400
arc=50 # adaptive representation consistency (semi-supervised)
akc=0 # adaptive knowledge consistency (transfer)
CUDA_VISIBLE_DEVICES=0,1 python -u main.py \
--data_root $data_root --dataset $dataset --num_labels $num_labels --pretrained_weight_path $pretrain_path  \
--lambda_mmd $arc --lambda_kd $akc 
```

默认的 MixMatch 参数是：
`--coef 500 --alpha 0.75 --alg ict --consistency ms --warmup_iter 4000 --ema_teacher true --ema_teacher_train true --ema_teacher_warmup true --ema_teacher_factor 0.999`

默认的 FixMatch 参数是： 
`--coef 0.5 --alg pl --strong_aug true --threshold 0.95  --ema_teacher true --ema_apply_wd true --ema_teacher_factor 0.999 --cutout_size 0.5`

4) 将训练和评估结果绘制为 [plot_loss.ipynb](plot_loss.ipynb)。
e.g: 在 CUB_200_2011 数据集上进行的实验中评估准确度，其中包含 400 个标记示例，如下所示。
![acc_curve](../figs/acc_curve.png)    

