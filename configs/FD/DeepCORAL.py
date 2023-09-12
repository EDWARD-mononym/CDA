configs = {
    "Dataset":{"Dataset_Name": "FD",
               "Scenarios": [("MFPT", "CWRU_DE", "PB_Real", "PB_Artificial", "CWRU_FE"), 
                             ("CWRU_DE", "PB_Real", "PB_Artificial", "CWRU_FE", "MFPT"), 
                             ("PB_Real", "PB_Artificial", "MFPT", "CWRU_DE", "CWRU_FE"), 
                             ("PB_Artificial", "MFPT", "CWRU_DE", "CWRU_FE", "PB_Real"), 
                             ("CWRU_FE", "CWRU_DE", "PB_Artificial", "PB_Real", "MFPT")],
               "num_class": 3,
               "input_channel": 1},

    "AdaptationConfig": {"Method": "DeepCORAL"},

    "BackboneConfig": {"Backbone": "CNN",
                       "feature_length": 5,
                       "hidden_channels": 64,
                       "kernel_size": 3,
                       "stride": 1,
                       "dropout": 0.5,
                       "hidden_layers": 5},
    
    "ClassifierConfig": {"Classifier": "Classifier"},

    "TrainingConfigs": {"n_epoch": 100},

    "OptimiserConfig": {"lr": 0.0005, "momentum": 0.9}
}