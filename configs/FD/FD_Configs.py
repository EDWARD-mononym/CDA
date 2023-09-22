configs = {
    "Dataset":{"Dataset_Name": "FD",
               "Scenarios": [("CWRU_DE", "PB_Artificial", "PB_Real", "CWRU_FE"), 
                             ("PB_Artificial", "CWRU_DE", "CWRU_FE", "PB_Real"),
                             ("CWRU_FE", "PB_Real", "CWRU_DE", "PB_Artificial")],
               "num_class": 3,
               "input_channel": 1},
    "ClassifierConfig": {"Classifier": "Classifier", "input_size": 128 * 5},

    "TrainingConfigs": {"n_epoch": 40},

    "OptimiserConfig": {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001,
                        "step_size": 50, "gamma": 0.5},

    "hparams": {"DANN": {"learning_rate": 00.10, "da_loss_wt":0.05},
              "DeepCORAL": {"learning_rate": 00.10, "da_loss_wt":0.05},
               "NRC": {"epsilon": 1e-50, "da_loss_wt": 0.05},
               "SHOT": {'ent_loss_wt': 0.8467, 'im': 0.2983,'target_cls_wt': 1},

               },
}
configs["ClassifierConfig"]["input_size"] = configs["BackboneConfig"]["output_channels"] * configs["BackboneConfig"]["feature_length"]