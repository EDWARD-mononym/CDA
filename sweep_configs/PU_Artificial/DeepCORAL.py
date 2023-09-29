configs = {
    "Dataset":{"Dataset_Name": "PU_Artificial",
               "Scenarios": [("Normal", "Rotate", "Load", "Radial")],
               "num_class": 8,
               "input_channel": 1},

    "AdaptationConfig": {"Method": "DeepCORAL"},

    "BackboneConfig": {"Backbone": "CNN",
                       "feature_length": 5,
                       "kernel_size": 5,
                       "stride": 1,
                       "dropout": 0.5,
                       "output_channels": 128,
                       "hidden_layers": 5},
    
    "ClassifierConfig": {"Classifier": "Classifier", "input_size": 128*5},

    "TrainingConfigs": {"n_epoch": 40},

    "OptimiserConfig": {"momentum": 0.9, "weight_decay": 0.0001, 
                        "step_size": 50, "gamma": 0.5},

    "Hyperparameters": {"lr": [0.001, 0.01], "hidden_channels": [64, 128]}
}

configs["ClassifierConfig"]["input_size"] = configs["BackboneConfig"]["output_channels"] * configs["BackboneConfig"]["feature_length"]