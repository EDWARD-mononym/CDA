import os

from utils.get_loaders import get_loader

####### Info #######
#? This function loads the necesary dataloaders and create the save folders before calling the update function for the algorithm
#? For detailed information on how the model adapts, check the individual algorithm class in algorithms folder

def adapt(algo_class, target_name, scenario, configs, writer, device, loss_avg_meters, method):
    trg_loader = get_loader(configs.Dataset_Name, target_name, "train")
    src_loader = get_loader(configs.Dataset_Name, scenario[0], "train")

    save_path = os.path.join(os.getcwd(), f'adapted_models/{configs.Dataset_Name}/{configs.adaptation(method)["Method"]}/{scenario}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    algo_class.update(src_loader, trg_loader,
                      scenario, target_name, configs.Dataset_Name,
                      save_path, writer, device, loss_avg_meters)