configs = {
    "Dataset":{"Dataset_Name": "HHAR",
               "Scenarios": [("0", "1", "2", "3", "4", "5", "6", "7", "8")],
               "num_class": 6,
               "input_channel": 3},

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