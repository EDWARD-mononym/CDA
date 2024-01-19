
import os
import logging
from utils.create_logger import create_writer
from utils.load_models import load_source_model, load_target_model

SEED = 42
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from train.base_trainer import Abstract_train
# from train.base_evaluator import DomainEvaluator

class DomainTrainer(Abstract_train):
    def __init__(self, args):
        super(DomainTrainer, self).__init__(args)
        # Load configurations and algorithm-specific parameters.
        self.configs, self.sweep_parameters = self.load_configs()
        # self.configs.alpha = args.alpha #! MODIFIED


    def train_and_log_source_model(self, source_name, scenario):
        """
        Train the model on the source domain and log its performance.
        """
        # Initialize the chosen algorithm with the loaded configurations.
        self.algo = self.load_algorithm(self.configs)
        # Pre-train the model on the source domain.
        self.pretrain(source_name)
        # Load the trained model's feature extractor and classifier.
        self.algo.feature_extractor, self.algo.classifier = load_source_model(
            self.configs, self.algo.feature_extractor, self.algo.classifier, scenario, self.device
        )
        # Test the model's performance on all domains in the scenario.
        source_accs = self.evaluator.test_all_domain(self.algo)
        # Update the results matrix with the source domain's performance metrics.
        self.evaluator.update(source_name, source_accs)

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
            self.algo.feature_extractor, self.algo.classifier = load_target_model(
                self.configs, self.algo.feature_extractor, self.algo.classifier, scenario, target_name, self.args.algo, self.configs.chkpoint_type,
                self.device
            )
            # Test the adapted model's performance on all domains in the scenario.
            target_accs = self.evaluator.test_all_domain(self.algo)
            # Update the results matrix with the target domain's performance metrics.
            self.evaluator.update(target_name, target_accs)
            # Estimating the Source Risk
            self.evaluator.get_src_risk(self.algo, scenario[0], self.loss_avg_meters)

        # Calculate overall metrics for the entire scenario.
        self.evaluator.calc_metric()