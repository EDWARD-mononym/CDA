from collections import defaultdict
import importlib.util
import os
import pandas as pd
import torch

from architecture.classifier import Classifier

class BaseModel(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()

        #* Getting feature extractor
        feature_extractor_module = importlib.import_module(f"architecture.{configs.backbone}")
        feature_extractor_class = getattr(feature_extractor_module, configs.backbone)
        self.feature_extractor = feature_extractor_class(configs)
        self.feature_extractor_class = feature_extractor_class

        #* Initialising classifier
        self.classifier = Classifier(configs)

        #* Optimisers
        self.feature_optimiser = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=configs.alg_hparams["learning_rate"],
            weight_decay=configs.train_params["weight_decay"], 
            betas=(0.5, 0.99)
        )
        self.classifier_optimiser = torch.optim.Adam(
            self.classifier.parameters(),
            lr=configs.alg_hparams["learning_rate"],
            weight_decay=configs.train_params["weight_decay"], 
        )

        #* Learning rate schedulers
        self.feature_lr_sched = torch.optim.lr_scheduler.StepLR(self.feature_optimiser, 
                                                                step_size=configs.train_params['step_size'], 
                                                                gamma=configs.train_params['lr_decay']
                                                                )
        self.classifier_lr_sched = torch.optim.lr_scheduler.StepLR(self.classifier_optimiser, 
                                                                   step_size=configs.train_params['step_size'], 
                                                                   gamma=configs.train_params['lr_decay']
                                                                   )

        #* Losses
        self.task_loss = torch.nn.CrossEntropyLoss()

        #* Utils
        self.configs = configs
        self.algo_name = "Source"
        self.epoch_performance = pd.DataFrame(columns=['Domain', 'Epoch Number', 'Average acc'])

    def train_source(self, src_loader, source_id, save_path, test_loader=None):
        best_loss = float('inf')

        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.train_params["N_epochs"]):
            print(f"training on {source_id} epoch: {epoch + 1}/{self.configs.train_params['N_epochs']}")
            losses = defaultdict(float)
            for x, y in src_loader:
                x, y = x.to(self.configs.device), y.to(self.configs.device)

                #* Zero the gradients
                self.feature_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                #* Forward pass
                feature = self.feature_extractor(x)
                pred = self.classifier(feature)

                #* Compute loss
                loss = self.task_loss(pred, y)

                #* Backpropagation
                loss.backward()
                self.feature_optimiser.step()
                self.classifier_optimiser.step()

                #* Record loss values
                losses["loss"] += loss.item() / len(src_loader)

            #* Learning rate scheduler
            self.feature_lr_sched.step()
            self.classifier_lr_sched.step()

            #* Save the losses of the current epoch
            for key in losses:
                epoch_losses[key].append(losses[key])

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"feature_extractor_{source_id}.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"classifier_{source_id}.pt"))

            #* If the test_loader was given, test the performance of current epoch on the test domain
            if test_loader and (epoch+1) % 10 == 0:
                self.evaluate(test_loader, epoch, source_id)


        return epoch_losses

    def evaluate (self, test_loader, epoch, current_domain):
        #? Measure the performance of the current model
        #? Called right after an epoch

        feature_extractor = self.feature_extractor.to(self.configs.device)
        classifier = self.classifier.to(self.configs.device)

        #* Eval mode
        feature_extractor.eval()
        classifier.eval()

        acc_list = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(self.configs.device)
                y = y.view((-1)).long().to(self.configs.device)

                #* Forward pass
                features = feature_extractor(x)
                predict_logits = classifier(features)
                pred = predict_logits.argmax(dim=1)

                #* Compute accuracy
                acc = torch.eq(pred, y).sum().item() / len(y)
                acc_list.append(acc)

        avg_acc = torch.tensor(acc_list).mean().item() #* Average accuracy

        df = pd.DataFrame({'Domain': current_domain,
                           'Epoch Number': epoch,
                           'Average acc': avg_acc}, index=[len(self.epoch_performance)])

        self.epoch_performance = pd.concat([self.epoch_performance, df], axis=0)