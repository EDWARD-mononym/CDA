import importlib
import os
import torch
from torch.optim.lr_scheduler import StepLR

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

        # optimizer and scheduler
        self.feature_extractor_optimiser = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        # optimizer and scheduler
        self.classifier_optimiser = torch.optim.Adam(
            self.classifier.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        
        self.fe_lr_scheduler = StepLR(self.feature_extractor_optimiser, step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = StepLR(self.classifier_optimiser, step_size=configs.step_size, gamma=configs.gamma)

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
            acc_dict = evaluator.test_all_domain(self)
            if acc_dict[target_name] > best_acc:
                best_acc = acc_dict[target_name]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{target_name}_feature_best.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{target_name}_classifier_best.pt"))

            #* Log epoch acc
            evaluator.update_epoch_acc(epoch, target_name, acc_dict)

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

    def pretrain(self, train_loader, test_loader, source_name, save_path, device, evaluator):
        best_acc = -1.0
        print(f"Training source model")
        for epoch in range(self.n_epoch):
            print(f'Epoch: {epoch}/{self.n_epoch}')

            self.feature_extractor.to(device)
            self.classifier.to(device)
            self.feature_extractor.train()
            self.classifier.train()
            running_loss = 0
            for step, data in enumerate(train_loader):
                x, y = data[0], data[1]
                x, y = x.to(device), y.to(device)

                # Zero grads
                self.feature_extractor_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                # Forward pass
                pred = self.classifier(self.feature_extractor(x))

                # Loss
                loss = self.taskloss(pred, y)
                loss.backward()

                # Step
                self.feature_extractor_optimiser.step()
                self.classifier_optimiser.step()

                running_loss += loss.item()

            # Adjust learning rate
            self.fe_lr_scheduler.step()
            self.classifier_lr_scheduler.step()

            #* Save best model
            acc_dict = evaluator.test_all_domain(self)
            epoch_acc = acc_dict[source_name]
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))

            # Print average loss every 'print_every' steps
            if (epoch + 1) % self.configs.print_every == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Epoch ACC: {acc_dict[source_name]}")
            print("-" * 30)  # Print a separator for clarity

            #* Log epoch acc
            evaluator.update_epoch_acc(epoch, source_name, acc_dict)

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        raise NotImplementedError