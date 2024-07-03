from torch.utils.data import random_split, Dataset, DataLoader
from glob import glob
from PIL import Image
import numpy as np
import torch
from random import random
import os

class CustomSegmentationDataset(Dataset):
    def __init__(self, root, transformations = None, bad_quality=False, test=0, masks=True):
        if masks:
            self.im_paths = sorted(glob(f"{root}/png_images/*.png"))
            self.gt_paths = sorted(glob(f"{root}/png_masks/*.png"))
        else:
            self.im_paths = []
            for ext in [".png", ".jpg"]:
                self.im_paths += sorted(glob(f"{root}/*{ext}"))
        if test:
            self.im_paths = self.im_paths[:test]
            if masks:
                self.gt_paths = self.gt_paths[:test]
        if bad_quality:
            self.im_paths *= 2
            if masks:
                self.gt_paths *= 2

        self.transformations = transformations
        self.n_cls = 59
        self.bad_quality = bad_quality
        self.masks = masks
        
        if masks:
            assert len(self.im_paths) == len(self.gt_paths)
        
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        if self.masks:
            im, gt = self.get_im_gt(self.im_paths[idx], self.gt_paths[idx])
            if self.transformations:
                im, gt = self.apply_transformations(im, gt)
            return im, gt
        else:
            im = self.get_im_gt(self.im_paths[idx], None)
            if self.transformations:
                im = self.apply_transformations(im, None)
            return im
        
    def get_im_gt(self, im_path, gt_path):
        im = Image.open(im_path).convert("RGB")
        if self.bad_quality and random() > 0.5:
            random_value = random() 
            im.save(f"tmp/tmp{random_value:.8f}.jpeg", quality=15, format='JPEG')
            im = Image.open(f"tmp/tmp{random_value:.8f}.jpeg").convert("RGB")
            if os.path.exists(f"tmp/tmp{random_value:.8f}.jpeg"):
                os.remove(f"tmp/tmp{random_value:.8f}.jpeg")
        im = np.array(im)
        if self.masks:
            gt = Image.open(gt_path).convert("L")
            gt = np.array(gt)
            return im, gt
        else:
            return im

    def apply_transformations(self, im, gt):
        if self.masks:
            transformed = self.transformations(image = im, mask = gt)
            return transformed["image"], transformed["mask"]
        else:
            transformed = self.transformations(image = im)
            return transformed["image"]


def get_dls(root, transformations, bs, split = [0.9, 0.05, 0.05], ns = 4, bad_quality=False, test=0, masks=True):
    if not test:
        assert sum(split) == 1., "Sum of the split must be exactly 1"
        
        ds = CustomSegmentationDataset(root = root, transformations = transformations, bad_quality=bad_quality, masks=masks)
        n_cls = ds.n_cls
        
        tr_len = int(len(ds) * split[0])
        val_len = int(len(ds) * split[1])
        test_len = len(ds) - (tr_len + val_len)
        
        # Data split
        tr_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [tr_len, val_len, test_len])

        print(f"\nThere are {len(tr_ds)} number of images in the train set")
        print(f"There are {len(val_ds)} number of images in the validation set")
        print(f"There are {len(test_ds)} number of images in the test set\n")
        
        # Get dataloaders
        tr_dl  = DataLoader(dataset = tr_ds, batch_size = bs, shuffle = True, num_workers = ns)
        val_dl = DataLoader(dataset = val_ds, batch_size = bs, shuffle = False, num_workers = ns)
        test_dl = DataLoader(dataset = test_ds, batch_size = 1, shuffle = False, num_workers = ns)
        
        return tr_dl, val_dl, test_dl, n_cls
    else:
        ds = CustomSegmentationDataset(root = root, transformations = transformations, bad_quality=bad_quality, test=test, masks=masks)
        test_dl = DataLoader(dataset = ds, batch_size = 1, shuffle = False, num_workers = ns)
        return test_dl
