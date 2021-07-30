import cv2
import numpy as np
import torch


def label2rgb(alpha, img, mask):
    labeled_img = cv2.addWeighted(decode_segmap(mask).transpose(1, 2, 0).astype(int), alpha, img.astype(int), 1 - alpha,
                                  0)
    return torch.from_numpy(labeled_img.transpose(2, 0, 1))


def decode_segmap(image: np.array, num_classes: int = 5):
    label_colors = np.array([(255, 255, 255), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for cls_idx in range(0, num_classes):
        idx = image == cls_idx
        r[idx] = label_colors[cls_idx, 0]
        g[idx] = label_colors[cls_idx, 1]
        b[idx] = label_colors[cls_idx, 2]
    rgb = np.stack([r, g, b], axis=0)
    return rgb


def unnormalize(img, mean=0.6993, std=0.4158):
    img = img * std
    img = img + mean
    return img * 255.0
