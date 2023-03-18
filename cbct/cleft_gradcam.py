import os

import pandas as pd
import numpy as np

from tqdm import tqdm

import SimpleITK as sitk

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from nets.classification import CleftNet, CleftSegNet
from loaders.cleft_dataset import CleftDataset, CleftSegDataset
from transforms.volumetric import CleftEvalTransforms, CleftSegEvalTransforms

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import argparse

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from medcam import medcam


# from pytorch_grad_cam import GradCAM, \
#     ScoreCAM, \
#     GradCAMPlusPlus, \
#     AblationCAM, \
#     XGradCAM, \
#     EigenCAM, \
#     EigenGradCAM, \
#     LayerCAM, \
#     FullGrad

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def main(args):
    
    if args.seg_column is None:
        model = CleftNet().load_from_checkpoint(args.model)
    else:
        model = CleftSegNet().load_from_checkpoint(args.model)
        
    model = model.model
    model.eval()
    model.cuda()
    

    if(os.path.splitext(args.csv)[1] == ".csv"):   
        df_train = pd.read_csv(args.csv_train)
        df_test = pd.read_csv(args.csv)
    else:        
        df_train = pd.read_parquet(args.csv_train)
        df_test = pd.read_parquet(args.csv)

    use_class_column = False
    if args.class_column is not None and args.class_column in df_test.columns:
        use_class_column = True

    if use_class_column:

        unique_classes = np.sort(np.unique(df_train[args.class_column]))
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
        class_replace = {}
        for cn, cl in enumerate(unique_classes):
            class_replace[cl] = cn
        print(unique_classes, unique_class_weights, class_replace)

        df_test[args.class_column] = df_test[args.class_column].replace(class_replace)

        if args.seg_column is None:
            test_ds = CleftDataset(df_test, img_column=args.img_column, mount_point=args.mount_point, class_column=args.class_column, transform=CleftEvalTransforms(256))
        else:
            test_ds = CleftSegDataset(df_test, img_column=args.img_column, mount_point=args.mount_point, class_column=args.class_column, seg_column=args.seg_column, transform=CleftSegEvalTransforms(256))

    else:
        test_ds = CleftDataset(df_test, img_column=args.img_column, mount_point=args.mount_point, transform=CleftEvalTransforms(256))

    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)


    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    targets = None

    if args.target_class is not None:
        targets = [ClassifierOutputTarget(args.target_class)]
    
    target_layers = [model.layer4[-1]]
    # Construct the CAM object once, and then re-use it on many images:
    cam = medcam.inject(model, replace=True, backend=args.backend, layer='layer4', label=args.target_class)
    # GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    if args.seg_column is None:
        for batch, (X, Y) in pbar: 
            
            X = X.cuda()
            grayscale_cam = cam(X)

            img_fn = df_test.loc[batch][args.img_column]

            out_fn = os.path.join(args.out, img_fn)

            out_dir = os.path.dirname(out_fn)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            grayscale_cam = grayscale_cam.squeeze().cpu().numpy()
            out_img = sitk.GetImageFromArray(grayscale_cam)

            img = sitk.ReadImage(os.path.join(args.mount_point, img_fn))
            out_img.SetSpacing(img.GetSpacing())
            out_img.SetOrigin(img.GetOrigin())
            out_img.SetDirection(img.GetDirection())

            pbar.set_description("Writing: {out_fn}".format(out_fn=out_fn))
            sitk.WriteImage(out_img, out_fn)
    else:
        for batch, (X0, X1, Y) in pbar: 

            X = torch.cat([X0, X1], dim=1).cuda()
            
            grayscale_cam = cam(X)

            img_fn = df_test.loc[batch][args.img_column]

            out_fn = os.path.join(args.out, img_fn)

            out_dir = os.path.dirname(out_fn)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            grayscale_cam = grayscale_cam.squeeze().cpu().numpy()
            out_img = sitk.GetImageFromArray(grayscale_cam)

            img = sitk.ReadImage(os.path.join(args.mount_point, img_fn))
            out_img.SetSpacing(img.GetSpacing())
            out_img.SetOrigin(img.GetOrigin())
            out_img.SetDirection(img.GetDirection())

            pbar.set_description("Writing: {out_fn}".format(out_fn=out_fn))
            sitk.WriteImage(out_img, out_fn)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--csv_train', type=str, help='CSV file to compute class replace', required=True)
    parser.add_argument('--extract_features', type=bool, help='Extract the features', default=False)
    parser.add_argument('--img_column', type=str, help='Column name in the csv file with image path', default="img")
    parser.add_argument('--class_column', type=str, help='Column name in the csv file with classes', default="Classification")
    parser.add_argument('--target_class', type=int, help='Class target number for gradcam', default=None)
    parser.add_argument('--seg_column', type=str, help='Column name in the csv file with image segmentation path', default=None)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--pred_column', help='Output column name', type=str, default="pred")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=1)
    parser.add_argument('--backend', type=str, default='gcam', help='Type of backend for medcam')
    parser.add_argument('--base_encoder', type=str, default='efficientnet-b0', help='Type of base encoder')

    args = parser.parse_args()

    main(args)