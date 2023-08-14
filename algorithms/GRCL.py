from collections import defaultdict
import itertools
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision import transforms

from algorithms.BaseModel import BaseModel
from architecture.MLP import MLP
from architecture.Losses import InfoNCE_loss
from utils import KMeans, combine_dataloaders

import time

class GRCL(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

        #* Feature encoder
        self.g = MLP(configs)
        self.g_optimiser = torch.optim.Adam(
            self.g.parameters(),
            lr=configs.alg_hparams["learning_rate"],
            weight_decay=configs.train_params["weight_decay"], 
            betas=(0.5, 0.99)
        )

        #* Data augmentation
        self.augment = transforms.Compose([
            transforms.ColorJitter(),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomHorizontalFlip()
        ])

        #* Optimizer
        self.u = torch.ones([2,1], requires_grad=True, device=configs.device)
        self.u_optimizer = torch.optim.LBFGS([self.u])

        #* Memory
        self.M = torch.utils.data.DataLoader(EmptyDataset(), batch_size=configs.train_params["batch_size"]) #? Stores episodic target memory
        self.B = torch.utils.data.DataLoader(EmptyDataset(), batch_size=configs.train_params["batch_size"]) #? Stores encoded features 
        self.CombinedLoader = torch.utils.data.DataLoader(EmptyDataset(), batch_size=configs.train_params["batch_size"]) #? Stores the loader we use which includes source, memory & target

        self.contrastive_loss = InfoNCE_loss
        self.KMean = KMeans(configs.num_classes, configs.features_len, device=configs.device)

    def update(self, src_loader, trg_loader, target_id, save_path, test_loader=None):

        best_loss = float('inf')
        epoch_losses = defaultdict(list) #* y axis datas to be plotted


        for epoch in range(self.configs.train_params["N_epochs"]):

            loss = self.train_epoch(src_loader, trg_loader, target_id, save_path, test_loader)


    def train_epoch(self, src_loader, trg_loader, target_id, save_path, test_loader=None):

        #* Set to train
        self.feature_extractor.train()
        self.g.train()

        self.CombinedLoader = combine_dataloaders(src_loader, self.M, trg_loader, batch_size=self.configs.train_params["batch_size"])

        #? if M is still empty, use a dataloader which returns None instead
        if len(self.M) == 0:
            NoneDataLoader = torch.utils.data.DataLoader(NoneDataset(self.configs.train_params["batch_size"]), 
                                                         batch_size=self.configs.train_params["batch_size"],
                                                         collate_fn = none_collate)
            joint_loader = enumerate(zip(itertools.cycle(src_loader), NoneDataLoader, self.CombinedLoader))

        else:
            joint_loader = enumerate(zip(itertools.cycle(src_loader), self.M, self.CombinedLoader))

        #* Loop through the joint dataset
        for step, ((src_x, src_y), (mem_x, mem_y), (trg_x, _)) in joint_loader:

            start = time.time()

            src_x, src_y, trg_x = src_x.to(self.configs.device), src_y.to(self.configs.device), trg_x.to(self.configs.device)

            self.feature_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            self.g_optimiser.zero_grad()

            #* Contrastive loss
            feature = self.feature_extractor(trg_x)
            encoded_feature = self.g(feature)
            with torch.no_grad():
                augmented_x = [self.augment(img) for img in trg_x]
                augmented_x = torch.stack(augmented_x)
                augmented_x = augmented_x.to(self.configs.device)
                k_plus = self.g(self.feature_extractor(augmented_x))

            contrast_loss = self.contrastive_loss(encoded_feature, k_plus, tau = self.configs.alg_hparams["tau"])
            contrast_loss.backward()
            self.g_optimiser.step()
            g_t = self.get_grad()

            # Reset gradient
            self.feature_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            self.g_optimiser.zero_grad()

            #* Domain classification loss
            pred_src = self.classifier(self.feature_extractor(src_x))
            src_loss = self.task_loss(pred_src, src_y)
            src_loss.backward()
            g_s = self.get_grad()

            # Reset gradient
            self.feature_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            self.g_optimiser.zero_grad()


            #* Memory classification loss
            if mem_x is None: # This means that m is empty and we're still on our first target domain
                g_dm = g_s #? Duplicate constraint which is equivalent to removing target memory constraint

            else:
                mem_x, mem_y = mem_x.to(self.configs.device), mem_y.to(self.configs.device)
                pred_mem = self.classifier(self.feature_extractor(mem_x))
                mem_loss = self.task_loss(pred_mem, mem_y)
                mem_loss.backward()
                g_dm = self.get_grad()

            # Reset gradient
            self.feature_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            self.g_optimiser.zero_grad()

            #* Optimise U & calculate optimal gradient update
            u_optimal = self.optimise(g_t, g_s, g_dm)
            g_optimal = (self.G.t() @ u_optimal).squeeze() + g_t 

            #* Update feature extractor
            #! too slow
            parameters_vector = torch.cat([param.view(-1) for param in self.feature_extractor.parameters()])
            gradients_vector = torch.cat([grad.view(-1) for grad in g_optimal])
            parameters_vector.grad = gradients_vector

            self.feature_optimiser.step()

            end = time.time()

            print(f"Time taken: {end - start}")


    def get_grad(self): 
        num_parameters = sum(p.numel() for p in self.feature_extractor.parameters())
        gradient_vector = torch.empty(num_parameters, device=self.configs.device)

        #* Populate the tensor with the gradients using enumerate
        index = 0
        for param in self.feature_extractor.parameters():
            size = param.numel()
            gradient_vector[index:index+size] = param.grad.view(-1)
            index += size

        return gradient_vector


    def UpdateMem(self, domain_loader):
        combined_loader = combine_dataloaders(self.M, domain_loader) #! This is not correct because we need to only select the top 1024

    def BuildFeatureBank(self, combined_loader): #? Initialise the feature bank
        new_b = None
        for x,_ in combined_loader:
            feature = self.feature_extractor(x)
            encoded_feature = self.g(feature)

    def UpdateKey(self):
        pass

    def optimise(self, g_t, g_s, g_dm):
        self.G = torch.cat([torch.neg(g_s).unsqueeze(0), torch.neg(g_dm).unsqueeze(0)], dim=0)
        self.g_t = g_t
        self.u_optimizer.step(self.closure)
        return self.u.detach()

    def closure(self):
        self.u_optimizer.zero_grad()
        loss =  self.objective(self.u)
        loss.backward()
        return loss

    def objective(self, u):
        term1 = 0.5 * u.t() @ self.G @ self.G.t() @ u
        term2 = self.g_t @ self.G.t() @ u
        return term1 - term2


    # @staticmethod
    # def objective(v, G, g_t):
    #     u = torch.exp(v) #? Enforce u > 0 by minimising v where u = exp(v)
    #     term1 = 0.5 * u.t() @ G @ G.t() @ u
    #     term2 = g_t @ G.t() @ u
    #     return term1 - term2

###### Used class & functions ######

class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0
    
    def __getitem__(self, index):
        raise IndexError("This dataset is empty!")

class NoneDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return None, None

def none_collate(batch):
    return None, None