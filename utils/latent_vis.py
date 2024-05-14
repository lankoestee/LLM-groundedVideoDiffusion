import torch
import torchvision
import matplotlib.pyplot as plt

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
        print(show_map.shape)
        img = torchvision.transforms.ToPILImage()(show_map)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

bear = torch.load("tmp/bear.pt", map_location='cpu')
single_vis(bear)