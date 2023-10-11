import importlib
import os
import torch

# from utils.model_testing import test_all_domain

####### Info #######
#? When creating a new algorithm, make the algo a subclass of BaseAlgo and redefine pretrain & epoch_train
#? src_loader will not be used if sourcefree
#? writer is a tensorboard to log the training process
#? configs should include parameters used which varies from algorithm to algorithm

class BaseAlgo(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        #* Import the feature extractor & classifier 
        backbone_name, classifier_name = configs.Backbone_Type , configs.Classifier_Type
        imported_backbone = importlib.import_module(f"architecture.{backbone_name}")
        imported_classifier = importlib.import_module(f"architecture.{classifier_name}")
        backbone_class = getattr(imported_backbone, backbone_name)
        classifier_class = getattr(imported_classifier, classifier_name)

        self.feature_extractor = backbone_class(configs)
        self.classifier = classifier_class(configs)

        self.n_epoch = configs.n_epoch

        self.configs = configs


    def update(self, src_loader, trg_loader,
               scenario, target_name, datasetname,
               save_path, writer, device, evaluator, loss_avg_meters):
        best_acc = -1.0
        print(f"Adapting to {target_name}")
        for epoch in range(self.n_epoch):
            print(f"Epoch: {epoch}/{self.n_epoch}")

            # Adaptation depends on the algorithm
            loss_dict = self.epoch_train(src_loader, trg_loader, epoch, device)

            # Test & Save best model

            acc_dict = evaluator.test_all_domain()
            if acc_dict[target_name] > best_acc:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{target_name}_feature_best.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{target_name}_classifier_best.pt"))

            # Log the performance of each domain for this epoch
            for domain in acc_dict:
                writer.add_scalar(f'Acc/{domain}', acc_dict[domain], epoch)

            # Log the losses
            for loss_name in loss_dict:
                loss_avg_meters[loss_name].update(loss_dict[loss_name])


            # Print average loss every 'print_every' steps
            if (epoch + 1) % self.configs.print_every == 0:
                for loss_name, loss_value in loss_dict.items():
                    loss_avg_meters[loss_name].update(loss_value)
                    print(f"{loss_name}: {loss_value:.4f}")
            print("-" * 30)  # Print a separator for clarity

        torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{target_name}_feature_last.pt"))
        torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{target_name}_classifier_last.pt"))

    def pretrain(self, train_loader, test_loader, save_path, device):
        raise NotImplementedError

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        raise NotImplementedError