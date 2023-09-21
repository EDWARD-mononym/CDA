import os

from utils.get_loaders import get_loader

####### Info #######
#? This function loads the necesary dataloaders and create the save folders before calling the update function for the algorithm
#? For detailed information on how the model adapts, check the individual algorithm class in algorithms folder

def adapt(algo_class, target_name, scenario, configs, writer, device):
    trg_loader = get_loader(configs["Dataset"]["Dataset_Name"], target_name, "train")
    src_loader = get_loader(configs["Dataset"]["Dataset_Name"], scenario[0], "train")

    save_path = os.path.join(os.getcwd(), f'adapted_models/{configs["Dataset"]["Dataset_Name"]}/{configs["AdaptationConfig"]["Method"]}/{scenario}')

    algo_class.update(src_loader, trg_loader,
                      scenario, target_name, configs["Dataset"]["Dataset_Name"],
                      save_path, writer, device)