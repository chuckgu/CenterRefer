import random

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
import skimage


def resize_and_pad(im, input_h, input_w):
    # Resize and pad im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = min(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[pad_h:pad_h+resized_h, pad_w:pad_w+resized_w, ...] = resized_im

    return new_im

def resize_and_crop(im, input_h, input_w):
    # Resize and crop im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = max(input_h / im_h, input_w / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    crop_h = int(np.floor(resized_h - input_h) / 2)
    crop_w = int(np.floor(resized_w - input_w) / 2)

    resized_im = skimage.transform.resize(im, [resized_h, resized_w])
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
    new_im[...] = resized_im[crop_h:crop_h+input_h, crop_w:crop_w+input_w, ...]

    return new_im


class FixScalePad:
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, h=320,w=320):
        self.h = h
        self.w = w

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample["image"]
        mask = sample["label"]
        text = sample["text"]
        center = sample["center"]
        img = np.array(img).astype(np.float32)
        text = np.array(text).astype(np.float32)
        center = np.array(center).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img = resize_and_pad(img,self.h,self.w)
        mask = resize_and_pad(mask,self.h,self.w)
        center = resize_and_pad(center,self.h,self.w)
        mask=(mask > 0)

        return {"image": img, "label": mask, "text": text, "center": center}

class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        text=sample["text"]
        center=sample["center"]
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img, "label": mask, "text": text, "center": center}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample["image"]
        mask = sample["label"]
        text=sample["text"]
        center=sample["center"]
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        text = np.array(text).astype(np.float32)
        center = np.array(center).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        center=center/np.max(center)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        center = torch.from_numpy(center).float()
        text = torch.from_numpy(text).float()

        return {"image": img, "label": mask, "text": text, "center": center}


class RandomHorizontalFlip:
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        text=sample["text"]
        center=sample["center"]
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img, "label": mask, "text": text, "center": center}


class RandomGaussianBlur:
    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        text=sample["text"]
        center=sample["center"]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {"image": img, "label": mask, "text": text, "center": center}


class RandomScaleCrop:
    def __init__(self, base_size, crop_size, fill=255):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        text=sample["text"]
        center=sample["center"]
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {"image": img, "label": mask, "text": text, "center": center}


class FixScale:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample["image"]
        mask = sample["label"]
        text=sample["text"]
        center=sample["center"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {"image": img, "label": mask, "text": text, "center": center}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        text=sample["text"]
        center=sample["center"]
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask, "text": text, "center": center}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        text=sample["text"]
        center=sample["center"]
        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask, "text": text, "center": center}