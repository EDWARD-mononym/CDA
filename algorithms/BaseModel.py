import torch
from collections import defaultdict
import os

from architecture.CNN import CNN
from architecture.classifier import Classifier

class BaseModel(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()

        #* Architectures & Backbone
        self.feature_extractor = CNN(configs)
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

        self.configs = configs
        self.algo_name = "Source"

    def train_source(self, src_loader):
        best_loss = float('inf')

        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.train_params["N_epochs"]):
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
                # torch.save(self.feature_extractor.state_dict(), os.path.join(self.configs.saved_models_path, self.algo_name, "feature_extractor_0.pt"))
                # torch.save(self.classifier.state_dict(), os.path.join(self.configs.saved_models_path, self.algo_name, "classifier_0.pt"))

        return epoch_losses

    def load_source_model(self):
        #* Initialise models
        source_feature_extractor = CNN(self.configs)
        source_classifier = Classifier(self.configs)

        #* Load state dicts
        source_feature_extractor.load_state_dict(torch.load(os.path.join(self.configs.saved_models_path, self.algo_name, "feature_extractor_0.pt")))
        source_classifier.load_state_dict(torch.load(os.path.join(self.configs.saved_models_path, self.algo_name, "classifier_0.pt")))

        #* Set to evaluation mode
        source_feature_extractor.eval()
        source_classifier.eval()

        return source_feature_extractor, source_classifier