import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets.classification import CleftNet, CleftSegNet
from loaders.cleft_dataset import CleftDataModule, CleftSegDataModule
from transforms.volumetric import CleftTrainTransforms, CleftEvalTransforms, CleftSegTrainTransforms, CleftSegEvalTransforms
from callbacks.logger import CleftImageLogger, CleftSegImageLogger

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)
        df_test = pd.read_parquet(args.csv_test)
    
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
    

    class_replace = {}
    for cn, cl in enumerate(unique_classes):
        class_replace[cl] = cn
    print(unique_classes, unique_class_weights, class_replace)
    
    df_train[args.class_column] = df_train[args.class_column].replace(class_replace)
    df_val[args.class_column] = df_val[args.class_column].replace(class_replace)
    df_test[args.class_column] = df_test[args.class_column].replace(class_replace)
    
    if args.seg_column is None:
        cleftdata = CleftDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, train_transform=CleftTrainTransforms(256), valid_transform=CleftEvalTransforms(256), test_transform=CleftEvalTransforms(256))

        model = CleftNet(args, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=args.base_encoder)

        image_logger = CleftImageLogger()
    else:

        cleftdata = CleftSegDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, train_transform=CleftSegTrainTransforms(256), valid_transform=CleftSegEvalTransforms(256), test_transform=CleftSegEvalTransforms(256))

        model = CleftSegNet(args, num_classes=unique_classes.shape[0], class_weights=unique_class_weights, base_encoder=args.base_encoder)

        image_logger = CleftSegImageLogger()
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model, datamodule=cleftdata, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Cleft classification Training')
    parser.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    parser.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    parser.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    parser.add_argument('--img_column', type=str, default='img', help='Name of image column')
    parser.add_argument('--class_column', type=str, default='Classification', help='Name of class column')
    parser.add_argument('--seg_column', type=str, default=None, help='Name of segmentation column')
    parser.add_argument('--base_encoder', type=str, default='efficientnet-b0', help='Type of base encoder')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model', help='Model path to continue training', type=str, default=None)
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=50)    
    parser.add_argument('--out', help='Output', type=str, default="./")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=50)
    
    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    parser.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification")


    args = parser.parse_args()

    main(args)
