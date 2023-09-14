configs = {
    "Dataset":{"Dataset_Name": "FD",
               "Scenarios": [("CWRU_DE", "PB_Artificial", "PB_Real", "CWRU_FE"), 
                             ("PB_Artificial", "CWRU_DE", "CWRU_FE", "PB_Real"),
                             ("CWRU_FE", "PB_Real", "CWRU_DE", "PB_Artificial")],
               "num_class": 3,
               "input_channel": 1},

    "AdaptationConfig": {"Method": "DANN"},

    "BackboneConfig": {"Backbone": "CNN",
                       "feature_length": 5,
                       "hidden_channels": 64,
                       "kernel_size": 5,
                       "stride": 1,
                       "dropout": 0.5,
                       "output_channels": 128,
                       "hidden_layers": 5},
    
    "ClassifierConfig": {"Classifier": "Classifier", "input_size": 128*5},

    "TrainingConfigs": {"n_epoch": 40},

    "OptimiserConfig": {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001, 
                        "step_size": 50, "gamma": 0.5}
}

configs["ClassifierConfig"]["input_size"] = configs["BackboneConfig"]["output_channels"] * configs["BackboneConfig"]["feature_length"]