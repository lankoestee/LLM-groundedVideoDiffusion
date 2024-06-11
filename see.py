import torch
import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

def single_vis(feature_map, filename):
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
        plt.savefig(f"tmp/{filename}.png")

def remove_background(image, n_clusters=2):
    image = image.to('cpu').numpy()
    c, h, w = image.shape
    flattened_image = image.reshape(c, -1).T  # 展平图像

    # 使用K-Means聚类进行前景背景分离
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_image)
    labels = kmeans.labels_.reshape(h, w)

    # 计算每个聚类的类内方差
    cluster_variances = []
    for i in range(n_clusters):
        cluster_pixels = flattened_image[labels.flatten() == i]
        cluster_variance = np.var(cluster_pixels, axis=0).mean()
        cluster_variances.append(cluster_variance)
    
    # 假设类内差距最小的类别是背景
    background_label = np.argmin(cluster_variances)
    mask = (labels != background_label).astype(np.uint8)

    # 使用形态学操作去除噪声和填补空洞
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3))).astype(np.uint8)

    mask = torch.tensor(mask)
    image = torch.tensor(image)

    # 将mask扩展到与图像相同的通道数
    mask = mask.unsqueeze(0).expand_as(image)
    image = image * mask

    # 根据mask裁剪图像
    non_zero = mask[0].nonzero()
    h_min = non_zero[:, 0].min()
    h_max = non_zero[:, 0].max()
    w_min = non_zero[:, 1].min()
    w_max = non_zero[:, 1].max()

    image = image[:, h_min:h_max, w_min:w_max]
    mask = mask[:, h_min:h_max, w_min:w_max]

    return image, mask

for file in os.listdir("tmp"):
    if file.endswith(".pt"):
        latent = torch.load(f"tmp/{file}", map_location="cpu")
        latent, mask = remove_background(latent[0], 10)
        single_vis(latent, file[:-3])