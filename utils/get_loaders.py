import importlib

def get_loader(dataset_name, domain_name, dtype):
    imported_dataset = importlib.import_module(f"dataloaders.{dataset_name}")
    create_dataloader_function = getattr(imported_dataset, "create_dataloader")
    loader = create_dataloader_function(domain_name, dtype)
    return loader