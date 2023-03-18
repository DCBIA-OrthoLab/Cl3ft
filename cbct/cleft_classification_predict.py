import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.utils.data import DataLoader

from nets.classification import CleftNet, CleftSegNet
from loaders.cleft_dataset import CleftDataset, CleftSegDataset
from transforms.volumetric import CleftEvalTransforms, CleftSegEvalTransforms
from callbacks.logger import CleftImageLogger

from sklearn.utils import class_weight
from sklearn.metrics import classification_report

from tqdm import tqdm

import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    
    if args.seg_column is None:
        model = CleftNet().load_from_checkpoint(args.model)
    else:
        model = CleftSegNet().load_from_checkpoint(args.model)
        
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

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=4)

    with torch.no_grad():

        predictions = []
        probs = []
        features = []
        for idx, X in tqdm(enumerate(test_loader), total=len(test_loader)): 
            if use_class_column and args.seg_column is None:
                X, Y = X
            elif use_class_column and args.seg_column is not None:
                X0, X1, Y = X
                X = torch.cat([X0, X1], dim=1) 
            X = X.cuda().contiguous()   
            if args.extract_features:        
                pred, x_f = model(X)    
                features.append(x_f.cpu().numpy())
            else:
                pred = model(X)
            probs.append(pred.cpu().numpy())        
            predictions.append(torch.argmax(pred, dim=1).cpu().numpy())
            

    df_test[args.pred_column] = np.concatenate(predictions, axis=0)
    probs = np.concatenate(probs, axis=0)    

    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    if use_class_column:
        print(classification_report(df_test[args.class_column], df_test[args.pred_column]))

    ext = os.path.splitext(args.csv)[1]
    if(ext == ".csv"):
        df_test.to_csv(os.path.join(args.out, os.path.basename(args.csv).replace(".csv", "_prediction.csv")), index=False)
    else:        
        df_test.to_parquet(os.path.join(args.out, os.path.basename(args.csv).replace(".parquet", "_prediction.parquet")), index=False)

    

    pickle.dump(probs, open(os.path.join(args.out, os.path.basename(args.csv).replace(ext, "_probs.pickle")), 'wb'))

    if len(features) > 0:
        features = np.concatenate(features, axis=0)
        pickle.dump(features, open(os.path.join(args.mount_point, args.out, os.path.basename(args.csv).replace(ext, "_prediction.pickle")), 'wb'))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='CSV file for testing', required=True)
    parser.add_argument('--csv_train', type=str, help='CSV file to compute class replace', required=True)
    parser.add_argument('--extract_features', type=bool, help='Extract the features', default=False)
    parser.add_argument('--img_column', type=str, help='Column name in the csv file with image path', default="img")
    parser.add_argument('--class_column', type=str, help='Column name in the csv file with classes', default="Classification")
    parser.add_argument('--seg_column', type=str, help='Column name in the csv file with image segmentation path', default=None)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--out', help='Output directory', type=str, default="./")
    parser.add_argument('--pred_column', help='Output column name', type=str, default="pred")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('--base_encoder', type=str, default='efficientnet-b0', help='Type of base encoder')

    args = parser.parse_args()

    main(args)
