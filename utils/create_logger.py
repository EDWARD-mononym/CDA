import os
import shutil
from torch.utils.tensorboard import SummaryWriter

def create_writer(dataset, method, scenario, target_id):
    writer_path = os.path.join(os.getcwd(), f"logs/{dataset}/{method}/{scenario}/{target_id}")

    print(f"Creating writer in {writer_path}")
    # if os.path.exists(writer_path):
    #     flag = input(f"{writer_path} will be removed, input yes to continue:")
    #     if flag == "yes":
    #             shutil.rmtree(writer_path, ignore_errors=True)

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