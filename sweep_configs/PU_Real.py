from ray import tune
class Configs:
    def __init__(self):
        # Dataset Configurations
        self.Dataset_Name = "PU_Real"
        self.Scenarios = [("Normal", "Rotate", "Load", "Radial")]
        self.num_class = 5
        self.input_channel = 1

        # Backbone Configurations
        self.Backbone_Type = "CNN"
        self.feature_length = 5
        self.hidden_channels = 64
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.output_channels = 128
        self.hidden_layers = 5

        # Classifier Configurations
        self.Classifier_Type = "Classifier"
        self.input_size = self.output_channels * self.feature_length

        # Training Configurations
        self.n_epoch = 1

        # Optimiser Configurations
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.step_size = 50
        self.gamma = 0.5

        # Tuned params
        self.lr = 1e-4
        self.algo_configs = self.adaptation()

        # self.tune_configs = self.adaptation()
    def adaptation(self, method="SHOT"):
        methods = {
            "SHOT":
                {
                # "Method": "SHOT",
                "lr": {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
                "ent_loss_wt": {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
                "im": {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
                "target_cls_wt": {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
                },
            "DeepCORAL": {
                # "Method": "DeepCORAL",
                "lr": {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
                "coral_wt":  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
        },
            "NRC": {
                "Method": "NRC",
                "epsilon": 1e-5,
            },
            "COSDA": {
                "Method": "COSDA",
                "beta": 2,
                "reg_alpha": 0.1,
                "temp": 0.07,  # Assuming 'temperature' in your code corresponds to 'temperature_begin'
                "temperature_end": 0.07,
                # Additional temperature value, you might want to adjust how this is used in your code
                "conf_gate": 0.4,  # Assuming 'conf_gate' in your code corresponds to 'confidence_gate_begin'
                "confidence_gate_end": 0.7,
                # Additional confidence gate value, you might want to adjust how this is used in your code
                "only_mi": False  # The actual value (True or False) for only_mi; Replace with the appropriate value
            }
        }
        return methods.get(method, {})

# Create an instance of the Configs class
configs = Configs()
#
# # Accessing the configurations
# print(configs.Dataset_Name)
# print(configs.Backbone_Type)
# print(configs.input_size)
#
# # Accessing the adaptation configuration for different methods
# print(configs.adaptation("SHOT"))
# print(configs.adaptation("DeepCORAL"))
