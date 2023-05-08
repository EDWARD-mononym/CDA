import torch
from collections import defaultdict

from algorithms.BaseModel import BaseModel
from architecture.discriminator import Discriminator

class DANN(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

        #* Architectures & Backbone
        self.discriminator = Discriminator(configs)

        #* Optimisers
        self.discriminator_optimiser = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=configs.hparams["learning_rate"],
            weight_decay=configs.hparams["weight_decay"], 
            betas=(0.5, 0.99)
        )

        #* Learning rate schedulers
        self.discriminator_lr_sched = torch.optim.lr_scheduler.StepLR(self.discriminator_optimiser,
                                                                      step_size=configs.hparams['step_size'], 
                                                                      gamma=configs.hparams['lr_decay']
                                                                      )

        #* Losses
        self.domain_loss = torch.nn.BCELoss()

    def update(self, dataloader, timestep):
        best_loss = float('inf')
        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.hparams["N_epochs"]):
            joint_loader = enumerate(zip(dataloader[0], dataloader[timestep]))
            losses = defaultdict(float)
            for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
                # len_dataloader = math.ceil(len(dataloader[0]) / 32) #TODO: Needs explaination
                # p = float(step + epoch * len_dataloader) / self.configs.hparams["N_epochs"] + 1 / len_dataloader #TODO: Needs explaination
                # alpha = 2. / (1. + np.exp(-10 * p)) - 1 #TODO: Needs explaination

                src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

                #* Zero grad
                self.feature_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()
                self.discriminator_optimiser.zero_grad()

                #* Forward pass
                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)
                trg_feat = self.feature_extractor(trg_x)

                #* Domain predictions
                src_domain_preds = self.discriminator(src_feat)
                src_domain_labels = torch.zeros_like(src_domain_preds)
                trg_domain_preds = self.discriminator(trg_feat)
                trg_domain_lables = torch.ones_like(trg_domain_preds)

                #* Compute loss
                task_loss = self.task_loss(src_pred, src_y)
                domain_loss = self.domain_loss(src_domain_preds, src_domain_labels) + self.domain_loss(trg_domain_preds, trg_domain_lables)
                loss = self.configs.DANN_params["task_weight"] * task_loss + self.configs.DANN_params["domain_weight"] * domain_loss

                #* Backpropagation
                loss.backward()
                self.feature_optimiser.step()
                self.classifier_optimiser.step()
                self.discriminator_optimiser.step()

                #* Record loss values
                losses["loss"] += loss.item() / len(dataloader[0])
                losses["task_loss"] += task_loss.item() / len(dataloader[0])
                losses["domain_loss"] += domain_loss.item() / len(dataloader[0])

            #* Learning rate scheduler
            self.feature_lr_sched.step()
            self.classifier_lr_sched.step()
            self.discriminator_lr_sched.step()

            #* Save the losses of the current epoch
            for key in losses:
                epoch_losses[key].append(losses[key])

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), f"{self.configs.saved_models_path}/feature_extractor_{timestep}.pt")
                torch.save(self.classifier.state_dict(), f"{self.configs.saved_models_path}/classifier_{timestep}.pt")

    #* Adapt function keeps the classifier constant
    #! Should only be called after training source model
    def adapt(self, dataloader, timestep):
        best_loss = float('inf')

        source_feature_extractor, _ = self.load_source_model() #* Loading source model
        
        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.hparams["N_epochs"]):
            losses = defaultdict(float)
            for (src_x, _), (trg_x, _) in zip(dataloader[0], dataloader[timestep]):
                src_x, trg_x = src_x.to(self.device), trg_x.to(self.device)

                #* Zero grad
                self.feature_optimiser.zero_grad()
                self.discriminator_optimiser.zero_grad()

                #* Forward pass
                src_feat = source_feature_extractor(src_x)
                trg_feat = self.feature_extractor(trg_x)

                #* Domain predictions
                src_domain_preds = self.discriminator(src_feat)
                src_domain_labels = torch.zeros_like(src_domain_preds)
                trg_domain_preds = self.discriminator(trg_feat)
                trg_domain_lables = torch.ones_like(trg_domain_preds)

                #* Compute loss
                domain_loss = self.domain_loss(src_domain_preds, src_domain_labels) + self.domain_loss(trg_domain_preds, trg_domain_lables)
                loss = domain_loss

                #* Backpropagation
                loss.backward()
                self.feature_optimiser.step()
                self.discriminator_optimiser.step()

                #* Record loss values
                losses["loss"] += loss.item() / len(dataloader[0])

            #* Learning rate scheduler
            self.feature_lr_sched.step()
            self.discriminator_lr_sched.step()

            #* Save the losses of the current epoch
            for key in losses:
                epoch_losses[key].append(losses[key])

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), f"{self.configs.saved_models_path}/feature_extractor_{timestep}.pt")

#TODO: Need explaination
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None