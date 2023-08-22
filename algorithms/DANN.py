from collections import defaultdict
import numpy as np
import os
import torch

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
            lr=configs.alg_hparams["learning_rate"],
            weight_decay=configs.train_params["weight_decay"], 
            betas=(0.5, 0.99)
        )

        #* Learning rate schedulers
        self.discriminator_lr_sched = torch.optim.lr_scheduler.StepLR(self.discriminator_optimiser,
                                                                      step_size=configs.train_params['step_size'], 
                                                                      gamma=configs.train_params['lr_decay']
                                                                      )

        #* Losses
        self.domain_loss = torch.nn.CrossEntropyLoss()

        self.algo_name = "DANN"

    def update(self, src_loader, trg_loader, target_id, save_path, test_loader=None):

        best_loss = float('inf')
        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        for epoch in range(self.configs.train_params["N_epochs"]):

            print(f"training on {target_id} epoch: {epoch + 1}/{self.configs.train_params['N_epochs']}")

            #* Set to train
            self.feature_extractor.train()
            self.classifier.train()

            joint_loader = enumerate(zip(src_loader, trg_loader))
            losses = defaultdict(float) #* To record losses
            for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
                src_x, src_y, trg_x = src_x.to(self.configs.device), src_y.to(self.configs.device), trg_x.to(self.configs.device)

                p = float(step + epoch * len(src_loader)) / self.configs.train_params["N_epochs"] + 1 / len(src_loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                

                #* Zero gradient
                self.feature_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()
                self.discriminator_optimiser.zero_grad()

                #* Domain Labels
                domain_label_src = torch.ones(len(src_x)).to(self.configs.device)
                domain_label_trg = torch.zeros(len(trg_x)).to(self.configs.device)

                #* Forward pass
                src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                trg_feat = self.feature_extractor(trg_x)

                #* Task classification loss
                task_loss = self.task_loss(src_pred.squeeze(), src_y)

                #* Domain Classification loss
                # Source
                src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
                src_domain_pred = self.discriminator(src_feat_reversed)
                src_domain_loss = self.domain_loss(src_domain_pred, domain_label_src.long())

                # Target
                trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
                trg_domain_pred = self.discriminator(trg_feat_reversed)
                trg_domain_loss = self.domain_loss(trg_domain_pred, domain_label_trg.long())

                #* Calculate Loss
                domain_loss = src_domain_loss + trg_domain_loss
                loss = self.configs.alg_hparams["src_cls_loss_wt"] * task_loss + self.configs.alg_hparams["domain_loss_wt"] * domain_loss

                #* Backward propagation
                loss.backward()
                self.feature_optimiser.step()
                self.classifier_optimiser.step()
                self.discriminator_optimiser.step()

                #* Record loss values
                losses["loss"] += loss.item() / len(src_loader)
                losses["task_loss"] += task_loss.item() / len(src_loader)
                losses["domain_loss"] += domain_loss.item() / len(src_loader)

            #* Learning rate scheduler
            self.feature_lr_sched.step()
            self.classifier_lr_sched.step()

            #* Save the losses of the current epoch
            for key in losses:
                epoch_losses[key].append(losses[key])

            #* Saves the model with the best total loss
            if losses["loss"] < best_loss:
                best_loss = losses["loss"]
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"feature_extractor_{target_id}.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"classifier_{target_id}.pt"))

            #* If the test_loader was given, test the performance of current epoch on the test domain
            if test_loader and (epoch+1) % 10 == 0:
                self.evaluate(test_loader, epoch, target_id)

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None