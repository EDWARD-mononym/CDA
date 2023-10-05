
import os
import logging
from utils.create_logger import create_writer
from utils.load_models import load_source_model, load_best_model

SEED = 42
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from train.base_train_test import Abstract_train

class DomainTrainer(Abstract_train):
    def __init__(self, args):
        super(DomainTrainer, self).__init__(args)
        # Load configurations and algorithm-specific parameters.
        self.configs, self.sweep_parameters = self.load_configs()
        # Initialize the chosen algorithm with the loaded configurations.
        self.algo = self.load_algorithm(self.configs)

    def train_and_log_source_model(self, source_name, scenario):
        """
        Train the model on the source domain and log its performance.
        """
        # Pre-train the model on the source domain.
        self.pretrain(source_name)
        # Load the trained model's feature extractor and classifier.
        self.algo.feature_extractor, self.algo.classifier = load_source_model(
            self.configs, self.algo.feature_extractor, self.algo.classifier, scenario, self.device
        )
        # Test the model's performance on all domains in the scenario.
        source_accs = self.test_all_domain(scenario)
        # Update the results matrix with the source domain's performance metrics.
        self.result_matrix.update(source_name, source_accs)

    def adapt_to_target_domains(self, scenario):
        """
        Adapt the pre-trained model to each target domain in the scenario.
        """
        # Iterate over each target domain in the scenario.
        for target_name in scenario[1:]:
            # Create a writer for logging purposes.
            writer = create_writer(self.configs.Dataset_Name, self.args.algo, scenario, target_name)
            # Adapt the model to the current target domain.
            self.adapt(target_name, scenario, writer)
            # Close the writer after adaptation is complete.
            writer.close()

            # Load the best-performing model for the current target domain.
            self.algo.feature_extractor, self.algo.classifier = load_best_model(
                self.configs, self.algo.feature_extractor, self.algo.classifier, scenario, target_name, self.args.algo,
                self.device
            )
            # Test the adapted model's performance on all domains in the scenario.
            target_accs = self.test_all_domain(scenario)
            # Update the results matrix with the target domain's performance metrics.
            self.result_matrix.update(target_name, target_accs)

        # Calculate overall metrics for the entire scenario.
        self.result_matrix.calc_metric()
        # Calculates and logs overall metrics for the scenario
        self.calc_overall_metrics


