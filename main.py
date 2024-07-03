import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import ImageFile
import torch
import segmentation_models_pytorch as smp
import os
import shutil

from dataset import get_dls
from train import train
from support_functions import visualize
from visualize import Plot
from test import inference

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_start_method('spawn', True)

def main():
    #Prepare
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    root = "dataset"
    mean, std, im_h, im_w = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 256, 256
    trans = A.Compose( [A.Resize(im_h, im_w), A.augmentations.transforms.Normalize(mean = mean, std = std), ToTensorV2(transpose_mask = True) ])
    
    while True:
        try:
            train_skip = int(input("Skip train?\n 1 = yes / 0 = no\n"))
            break
        except:
            print("incorrect value, enter integer value.\n")

    if not train_skip:
        #Dataset
        while True:
            try:
                bq = int(input("Train bad quality images?\n 1 = yes / 0 = no\n"))
                break
            except:
                print("incorrect value, enter integer value.\n")
        tr_dl, val_dl, test_dl, n_cls = get_dls(root = root, transformations = trans, bs = 16, bad_quality=bq)
        visualize(tr_dl.dataset, n_ims = 20)

        #Model
        model = smp.DeepLabV3Plus(classes = n_cls)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = 3e-4)

        #Train
        history = train(model = model, tr_dl = tr_dl, val_dl = val_dl,
            loss_fn = loss_fn, opt = optimizer, device = device, n_cls=n_cls,
            epochs = 50, save_prefix = f"{'with_bad' if bq else 'no_bad'}")

        #Visualize
        Plot(history)

    #Test
    while True:
        try:
            model_type = int(input("Which model use?\n 1 = with_bad_quality / 0 = no_bad_quality\n"))
            break
        except:
            print("incorrect value, enter integer value.\n")
    model = torch.load(f"saved_models/{'with_bad' if model_type else 'no_bad'}_best_model.pt")
    
    while True:
        try:
            test_skip = int(input("Skip dataset test and go to your own images?\n 1 = yes / 0 = no\n"))
            break
        except:
            print("incorrect value, enter integer value.\n")
    if not test_skip:      
        test_dl = get_dls(root = root, transformations = trans, bs = 16, bad_quality=True, test=50)
        inference(test_dl, model = model, device = device)

    #Custom test
    while True:
        try:
            CT = int(input("Test own images?\n 1 = yes / 0 = no\n"))
            break
        except:
            print("incorrect value, enter integer value.\n")
    if CT:
        test_dl = get_dls(root = "custom_images", transformations = trans, bs = 16, bad_quality=False, test=50, masks=False)
        inference(test_dl, model = model, device = device, masks=False)

    #Exit
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
        os.mkdir("tmp")
    input("Enter to close...")

if __name__ == "__main__":
    main()