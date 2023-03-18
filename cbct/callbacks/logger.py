from pytorch_lightning.callbacks import Callback
import torchvision
import torch

class CleftImageLogger(Callback):
    def __init__(self, num_images=256, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            x, y = batch

            x = torch.permute(x[0], (1, 0, 2, 3)) # get the first image in the batch and permute the channels dimension with i dimension, leave j, k dimensions. 
            x = x - torch.min(x)
            x = x/torch.max(x)

            grid_x = torchvision.utils.make_grid(x[0:self.num_images])
            trainer.logger.experiment.add_image('x', grid_x, batch_idx)

class CleftSegImageLogger(Callback):
    def __init__(self, num_images=256, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            x0, x1, y = batch

            x = torch.permute(x0[0], (1, 0, 2, 3)) # get the first image in the batch and permute the channels dimension with i dimension, leave j, k dimensions. 
            x = x - torch.min(x)
            x = x/torch.max(x)

            grid_x = torchvision.utils.make_grid(x[0:self.num_images])
            trainer.logger.experiment.add_image('x', grid_x, batch_idx)

            

            x = torch.permute(x1[0], (1, 0, 2, 3)) # get the first image in the batch and permute the channels dimension with i dimension, leave j, k dimensions. 
            x = x - torch.min(x)
            x = x/torch.max(x)

            grid_x = torchvision.utils.make_grid(x[0:self.num_images])
            trainer.logger.experiment.add_image('seg', grid_x, batch_idx)