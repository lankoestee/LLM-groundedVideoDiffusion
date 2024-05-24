import torch
import torchvision
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from utils.layout_make import latent_embed

def single_vis(feature_map):
    """
    Visualize a single feature map
    :param feature_map: [B, C, H, W] or [B, T, C, H, W]
    """
    device = feature_map.device
    if device.type == 'cuda':
        feature_map = feature_map.cpu()
    if len(feature_map.shape) in [3, 4]:
        feature_map = feature_map.float()
        if len(feature_map.shape) == 4:
            show_map = feature_map[0]
            show_map = show_map.squeeze(0)
        else:
            show_map = feature_map
        # 在通道维度上取平均
        show_map = torch.mean(show_map, 0)
        # show_map = show_map[0]
        # 归一化到[0, 1]
        show_map = (show_map - show_map.min()) / (show_map.max() - show_map.min())
        img = torchvision.transforms.ToPILImage()(show_map)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def remove_background(image, edge_size=10):
    tensor = torch.mean(image[0], 0)
    tensor = tensor.squeeze(0)
    h, w = tensor.shape

    # 提取图像边缘的像素用于背景分析
    edges = np.concatenate([
        tensor[:edge_size, :].flatten(),
        tensor[-edge_size:, :].flatten(),
        tensor[:, :edge_size].flatten(),
        tensor[:, -edge_size:].flatten()
    ])

    # 计算背景像素的均值和标准差
    bg_mean = np.mean(edges)
    bg_std = np.std(edges)

    # 设置阈值，通常为背景均值±3倍的标准差
    threshold = bg_mean + 3 * bg_std

    # 创建mask，背景部分为0，主体部分为1
    mask = np.where(np.abs(tensor - bg_mean) <= threshold, 0, 1)

    # 使用形态学操作去除噪声和填补空洞
    mask = ndimage.binary_closing(mask, structure=np.ones((3,3))).astype(np.uint8)
    mask = ndimage.binary_opening(mask, structure=np.ones((3,3))).astype(np.uint8)

    mask = torch.tensor(mask)
    image[0] = image[0] * mask

    # 根据mask裁剪图像
    non_zero = mask.nonzero()
    h_min = non_zero[:, 0].min()
    h_max = non_zero[:, 0].max()
    w_min = non_zero[:, 1].min()
    w_max = non_zero[:, 1].max()

    image = image[:, :, h_min:h_max, w_min:w_max]
    mask = mask[h_min:h_max, w_min:w_max]

    return image, mask.expand_as(image)


bear = torch.load("tmp/bear.pt", map_location="cpu")
bear, bear_mask = remove_background(bear, 10)
noise = latent_embed(bear, fps=24)
noise = noise.permute(0, 2, 1, 3, 4)
single_vis(noise[0])