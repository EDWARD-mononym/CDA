configs = {
    "Dataset":{"Dataset_Name": "PU_Real",
               "Scenarios": [("Normal", "Rotate", "Load", "Radial")],
               "num_class": 5,
               "input_channel": 1},

    "AdaptationConfig": {"Method": "COSDA"},

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
                        "step_size": 50, "gamma": 0.5},

    "hparams" : {
                  "beta": 2,
                  "reg_alpha": 0.1,
                  "temp": 0.07,  # Assuming 'temperature' in your code corresponds to 'temperature_begin'
                  "temperature_end": 0.07,
                  # Additional temperature value, you might want to adjust how this is used in your code
                  "conf_gate": 0.4,  # Assuming 'conf_gate' in your code corresponds to 'confidence_gate_begin'
                  "confidence_gate_end": 0.7,
                  # Additional confidence gate value, you might want to adjust how this is used in your code
                  "only_mi": False # The actual value (True or False) for only_mi; Replace with the appropriate value
}
}

configs["ClassifierConfig"]["input_size"] = configs["BackboneConfig"]["output_channels"] * configs["BackboneConfig"]["feature_length"]