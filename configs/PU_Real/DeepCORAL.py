configs = {
    "Dataset":{"Dataset_Name": "PU_Real",
               "Scenarios": [("Normal", "Rotate", "Load", "Radial")],
               "num_class": 16,
               "input_channel": 1},

    "AdaptationConfig": {"Method": "DeepCORAL"},

    "BackboneConfig": {"Backbone": "CNN",
                       "feature_length": 5,
                       "hidden_channels": 64,
                       "kernel_size": 5,
                       "stride": 1,
                       "dropout": 0.5,
                       "output_channels": 128,
                       "hidden_layers": 5},
    
    "ClassifierConfig": {"Classifier": "Classifier", "input_size": 128*5},

    "TrainingConfigs": {"n_epoch": 2},

    "OptimiserConfig": {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001, 
                        "step_size": 50, "gamma": 0.5}
}

configs["ClassifierConfig"]["input_size"] = configs["BackboneConfig"]["output_channels"] * configs["BackboneConfig"]["feature_length"]