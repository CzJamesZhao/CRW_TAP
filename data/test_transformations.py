from __future__ import print_function, absolute_import

from glob import glob
import os.path as osp
import mediapy as media
from flow_augmentations import (CenterCrop, RandomResizedCrop, Resize,
                                     ToTensor)
import torch

random_seed=786234
data_root = "/home/zhaochenzhi/CRW_TAP/datasets/kinetics700-2020"
split = "train"

video_list = sorted(glob(osp.join(data_root, split, "*/*.mp4")))
# 使用第一个视频来构建transformations
video_id = video_list[0]

frames = media.read_video(video_id)[:2]

random_resized_crop_transform = RandomResizedCrop(
    size=(448, 448),
    scale=(0.3, 1.0),
    ratio=(0.75, 1.33),
    seed=random_seed,
)

# 转换为 PyTorch 张量
frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (T, C, H, W)，mediapy 输出为 (T, H, W, C)
frames = frames.float() / 255.0  # 归一化到 [0, 1]

# 正向和反向的增强处理
frames_forward = frames.clone()  # 前向帧
frames_backward = frames.flip(0)  # 时间维度翻转的帧

# 应用随机裁剪
_, _, _, affine_mat_forward = random_resized_crop_transform(frames_forward)
_, _, _, affine_mat_backward = random_resized_crop_transform(frames_backward)

# 比较 affine_mat_forward 和 affine_mat_backward
are_matrices_equal = torch.allclose(affine_mat_forward, affine_mat_backward, atol=1e-6)

# 输出结果
print("Affine matrices comparison:")
print(f"Affine matrix forward:\n{affine_mat_forward}")
print(f"Affine matrix backward:\n{affine_mat_backward}")
print(f"Are the matrices equal? {are_matrices_equal}")
