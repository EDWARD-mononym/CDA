configs = {
    "Dataset":{"Dataset_Name": "PU_Real",
               "Scenarios": [("Normal", "Rotate", "Load", "Radial")],
               "num_class": 5,
               "input_channel": 1},

    "AdaptationConfig": {"Method": "DANN"},

    "BackboneConfig": {"Backbone": "CNN"},
    
    "ClassifierConfig": {"Classifier": "Classifier"},

    "TrainingConfigs": {"n_epoch": 40},

    "OptimiserConfig": {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001, 
                        "step_size": 50, "gamma": 0.5},

    "Hyperparameters": {"lr": [0.001, 0.01], "momentum": [0.9], "weight_decay": [0.0001], "step_size": [50], "gamma": [0.5], # Optimiser parameters
                        "feature_length": [5], "hidden_channels": [64], "kernel_size": [5], "stride": [1], "dropout": [0.5], "output_channels": [128], "hidden_layers": [5], # Backbone parameters
                        'ent_loss_wt': [0.01,0.05,0.1, 0.5,1], 'im':[0.01,0.05,0.1, 0.5,1] ,'target_cls_wt': 1
                        }
}