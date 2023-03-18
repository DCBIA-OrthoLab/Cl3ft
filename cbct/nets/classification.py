import math
from typing import Optional, Tuple

import numpy as np 

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
import torchmetrics

import monai
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution

import pytorch_lightning as pl

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value).view(batch_size, -1, hidden_dim)
        context = torch.sum(context, dim=1)

        return context, attn

class SelfAttention(nn.Module):
    def __init__(self, in_units, out_units):
        super(SelfAttention, self).__init__()

        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(nn.Tanh()(self.W1(query)))

        # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
        # min_score = tf.reshape(min_score, [-1, 1, 1])
        # score_mask = tf.greater_equal(score, min_score)
        # score_mask = tf.cast(score_mask, tf.float32)
        # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

        # attention_weights shape == (batch_size, max_length, 1)
        score = nn.Sigmoid()(score)
        sum_score = torch.sum(score, 1, keepdim=True)
        attention_weights = score / sum_score

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class CleftNet(pl.LightningModule):
    def __init__(self, args = None, class_weights=None, base_encoder="efficientnet-b0", num_classes=3):
        super(CleftNet, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
        
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)

        if self.hparams.base_encoder == 'SEResNet50':
            self.model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=1, num_classes=self.hparams.num_classes)
        elif self.hparams.base_encoder == 'resnet18': 
            self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=self.hparams.num_classes)
        else:
            self.model = monai.networks.nets.EfficientNetBN(self.hparams.base_encoder, spatial_dims=3, in_channels=1, num_classes=self.hparams.num_classes)

        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)        

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        x = self(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self(x)

        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)


class CleftSegNet(pl.LightningModule):
    def __init__(self, args = None, class_weights=None, base_encoder="efficientnet-b0", num_classes=3):
        super(CleftSegNet, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
        
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)

        if self.hparams.base_encoder == 'SEResNet50':
            self.model = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=2, num_classes=self.hparams.num_classes)
        elif self.hparams.base_encoder == 'resnet18': 
            self.model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=2, num_classes=self.hparams.num_classes)
        else:
            self.model = monai.networks.nets.EfficientNetBN(self.hparams.base_encoder, spatial_dims=3, in_channels=2, num_classes=self.hparams.num_classes)

        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)        

    def training_step(self, train_batch, batch_idx):
        x0, x1, y = train_batch

        x = torch.cat([x0, x1], dim=1)

        x = self(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)
        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x0, x1, y = val_batch
        
        x = torch.cat([x0, x1], dim=1)
        x = self(x)

        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)