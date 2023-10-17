import os
import wandb
from torch.utils.tensorboard import SummaryWriter

def create_writer(dataset, method, scenario, target_id):

    if wandb.run is None:
        writer_path = os.path.join(os.getcwd(), f"logs/{dataset}/{method}/{scenario}/{target_id}")
    else:
        writer_path = wandb.run.dir
    print(f"Creating writer in {writer_path}")
    writer = SummaryWriter(log_dir=writer_path)

    return writer

class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count