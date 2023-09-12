import importlib

def get_loader(dataset_name, domain_name, dtype):
    imported_dataset = importlib.import_module(f"dataloaders.{dataset_name}")
    loader_name = f"{domain_name}_{dtype}" # dtype is "test" or "train"
    loader = getattr(imported_dataset, loader_name) #? Equivalent to "from dataloaders.{dataset_name} import loader_name"
    return loader