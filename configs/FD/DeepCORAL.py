configs = {
    "Dataset":{"Dataset_Name": "FD",
               "Scenarios": [("zero", "one", "two", "three")],
               "num_class": 4,
               "input_channel": 1},

    "AdaptationConfig": {"Method": "DeepCORAL"},

    "BackboneConfig": {"Backbone": "CNN",
                       "feature_length": 5,
                       "hidden_channels": 64,
                       "kernel_size": 3,
                       "stride": 1,
                       "dropout": 0.5,
                       "hidden_layers": 2},
    
    "ClassifierConfig": {"Classifier": "Classifier", "input_size": 64*5},

    "TrainingConfigs": {"n_epoch": 100},

    "OptimiserConfig": {"lr": 0.0005, "momentum": 0.9}
}