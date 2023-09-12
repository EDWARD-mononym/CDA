import os
import shutil
from torch.utils.tensorboard import SummaryWriter

def create_writer(dataset, method, scenario, target_id):
    writer_path = os.path.join(os.getcwd(), f"logs/{dataset}/{method}/{scenario}/{target_id}")

    print(f"Creating writer in {writer_path}")
    if os.path.exists(writer_path):
        flag = input(f"{writer_path} will be removed, input yes to continue:")
        if flag == "yes":
                shutil.rmtree(writer_path, ignore_errors=True)

    writer = SummaryWriter(log_dir=writer_path)

    return writer