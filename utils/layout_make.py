from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils.latent_vis import single_vis, remove_background
from diffusers.utils.torch_utils import randn_tensor

def image_embed(image, parsed_layout = None, H = 320, W = 576):
    if parsed_layout is None:
        # 原始布局
        original_layout = {
            0: (0.0, 0.5, 0.1953125, 0.6953125),
            1: (0.166015625, 0.5, 0.361328125, 0.6953125),
            2: (0.33203125, 0.5, 0.52734375, 0.6953125),
            3: (0.498046875, 0.5, 0.693359375, 0.6953125),
            4: (0.6640625, 0.5, 0.859375, 0.6953125),
            5: (0.830078125, 0.5, 1.025390625, 0.6953125)
        }

        # 原始布局中的键
        original_keys = list(original_layout.keys())

        # 将布局点转换为NumPy数组
        points = np.array([original_layout[key] for key in original_keys])

        # 新的布局键
        new_keys = np.linspace(original_keys[0], original_keys[-1], 24)

        # 插值函数
        def interpolate_layout(old_keys, old_points, new_keys):
            new_points = []
            for i in range(old_points.shape[1]):
                new_points.append(np.interp(new_keys, old_keys, old_points[:, i]))
            return np.array(new_points).T

        # 进行插值
        interpolated_points = interpolate_layout(original_keys, points, new_keys)

        # 将插值后的点转换为字典
        layouts = {i: tuple(interpolated_points[i]) for i in range(len(interpolated_points))}
    else:
        layouts = parsed_layout
    
    # 创造24帧Image高斯噪声，大小为[320*576]
    ret_images = []
    for i in range(24):
        background = np.zeros((H, W, 3))
        shape = ((layouts[i][3] - layouts[i][1]) * H, 
                 (layouts[i][2] - layouts[i][0]) * W)
        start = (int(layouts[i][1] * H), int(layouts[i][0] * W))
        shape = (int(shape[0]), int(shape[1]), image.shape[2])
        body = np.array(Image.fromarray(image).resize((shape[1], shape[0])))
        for h in range(shape[0]):
            for w in range(shape[1]):
                if body[h, w, 3] != 0:
                    if start[0] + h < H and start[1] + w < W:
                        background[start[0] + h, start[1] + w] = body[h, w, :3]
        ret_images.append(background)
    return ret_images

def latent_embed(latent, parsed_layout=None, fps=24, H=40, W=72, generator=None):
    assert len(latent.shape) in [3, 4]
    if len(latent.shape) == 3:
        C, _, _ = latent.shape
        image = latent
    elif len(latent.shape) == 4:
        _, C, _, _ = latent.shape
        image = latent[0]
    else:
        raise ValueError("Invalid latent shape")
    
    if parsed_layout is None:
        # 原始布局
        original_layout = {
            0: (0.0, 0.5, 0.1953125, 0.6953125),
            1: (0.166015625, 0.5, 0.361328125, 0.6953125),
            2: (0.33203125, 0.5, 0.52734375, 0.6953125),
            3: (0.498046875, 0.5, 0.693359375, 0.6953125),
            4: (0.6640625, 0.5, 0.859375, 0.6953125),
            5: (0.830078125, 0.5, 1.025390625, 0.6953125)
        }

        # 原始布局中的键
        original_keys = list(original_layout.keys())

        # 将布局点转换为NumPy数组
        points = np.array([original_layout[key] for key in original_keys])

        # 新的布局键
        new_keys = np.linspace(original_keys[0], original_keys[-1], fps)

        # 插值函数
        def interpolate_layout(old_keys, old_points, new_keys):
            new_points = []
            for i in range(old_points.shape[1]):
                new_points.append(np.interp(new_keys, old_keys, old_points[:, i]))
            return np.array(new_points).T

        # 进行插值
        interpolated_points = interpolate_layout(original_keys, points, new_keys)

        # 将插值后的点转换为字典
        layouts = {i: tuple(interpolated_points[i]) for i in range(len(interpolated_points))}
    else:
        layouts = parsed_layout
    
    noise = randn_tensor((1, fps, C, H, W), generator=generator, dtype=torch.float16, device=latent.device)
    for i in range(fps):
        background = noise[0, i]
        shape = ((layouts[i][3] - layouts[i][1]) * H, 
                 (layouts[i][2] - layouts[i][0]) * W)
        start = (int(layouts[i][1] * H), int(layouts[i][0] * W))
        shape = (int(shape[0]), int(shape[1]), C)
        body = image[:, start[0]:start[0]+shape[0], start[1]:start[1]+shape[1]]
        for h in range(shape[0]):
            for w in range(shape[1]):
                if body[:, h, w].sum() != 0:
                    if start[0] + h < H and start[1] + w < W:
                        background[:, start[0] + h, start[1] + w] = body[:, h, w]
        noise[0, i] = background
    noise = noise.permute(0, 2, 1, 3, 4)
    noise = randn_tensor((1, fps, C, H, W), generator=generator, dtype=torch.float16, device=latent.device)
    return noise


def png_image_process(image):
    shape = image.shape
    # 将所有透明度为0的像素点的RGB值设为0
    for h in range(shape[0]):
        for w in range(shape[1]):
            if image[h, w, 3] == 0:
                image[h, w, :3] = 0
    # 删除透明度通道
    image = image[:, :, :3]
    return image
