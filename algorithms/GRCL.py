from collections import defaultdict
import heapq
import itertools
import os
import torch
from torchvision import transforms
from tqdm import tqdm

from algorithms.BaseModel import BaseModel
from architecture.MLP import MLP
from architecture.Losses import InfoNCE_loss
from utils import KMeans, combine_dataloaders

class GRCL(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)

        #* Projection layer
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
        self.v = torch.ones([2,1], requires_grad=True, device=configs.device)
        self.v_optimizer = torch.optim.LBFGS([self.v])

        #* Memory
        self.M = torch.utils.data.DataLoader(EmptyDataset(), batch_size=configs.train_params["batch_size"]) #? Stores episodic target memory
        self.B = torch.utils.data.DataLoader(EmptyDataset(), batch_size=configs.train_params["batch_size"]) #? Stores encoded features 

        self.contrastive_loss = InfoNCE_loss
        self.KMean = KMeans(configs.alg_hparams["K"], configs.features_len * configs.final_out_channels, device=configs.device)

    def update(self, src_loader, trg_loader, target_id, save_path, test_loader=None):

        best_loss = float('inf')
        epoch_losses = defaultdict(list) #* y axis datas to be plotted

        combined_domain = combine_dataloaders(src_loader, self.M, trg_loader) #? D_{s} U M U D_{t}
        self.BuildFeatureBank(combined_domain)

        for epoch in range(self.configs.train_params["N_epochs"]):

            print(f"training on {target_id} epoch: {epoch + 1}/{self.configs.train_params['N_epochs']}")

            loss = self.train_epoch(src_loader)

            #* Learning rate scheduler
            self.feature_lr_sched.step()

            #* Saves the model with the best total loss
            if loss < best_loss:
                best_loss = loss
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"feature_extractor_{target_id}.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"classifier_{target_id}.pt"))

            #* If the test_loader was given, test the performance of current epoch on the test domain
            if test_loader and (epoch+1) % 10 == 0:
                self.evaluate(test_loader, epoch, target_id)

        self.UpdateMem(trg_loader) # Saves top predictors from this target domain into memory

    def train_epoch(self, src_loader):

        #* Set to train
        self.feature_extractor.train()
        self.g.train()

        losses = defaultdict(float) #* To record losses

        if len(self.M) == 0:
            NoneDataLoader = torch.utils.data.DataLoader(NoneDataset(self.configs.train_params["batch_size"]), 
                                                         batch_size=self.configs.train_params["batch_size"],
                                                         collate_fn = none_collate)
            joint_loader = enumerate(zip(itertools.cycle(src_loader), itertools.cycle(NoneDataLoader), self.B_loader))

        else:
            joint_loader = enumerate(zip(itertools.cycle(src_loader), itertools.cycle(self.M), self.B_loader))

        #* Loop through the joint dataset
        for step, ((src_x, src_y), (mem_x, mem_y), (trg_x, _, idx)) in joint_loader:

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

            self.UpdateKey(trg_x, idx)

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
                mem_loss = torch.zeros(1)

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

            params = list(self.feature_extractor.parameters())
            for i in range(len(params)):
                if params[i].grad is not None:
                    params[i].grad.copy_(g_optimal[i])

            self.feature_optimiser.step()

            #* Record total src + prev target loss
            losses["loss"] += (src_loss.item() + mem_loss.item()) / self.configs.train_params["batch_size"]

        return losses["loss"]

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
        # Optimize & label centroids
        joint_loader = combine_dataloaders(self.M, domain_loader, batch_size=self.configs.train_params["batch_size"])
        self.KMean.optimize(joint_loader, self.feature_extractor, self.classifier)

        #* Store the top 1024 data and pseudo labels
        top_data = []
        for x, _ in domain_loader:
            x = x.to(self.configs.device)
            with torch.no_grad():
                feature = self.feature_extractor(x)
                pred = self.classifier(feature)
            pseudo_labels = self.KMean.generate_pseudolabel(feature)
            logits = pred[torch.arange(len(x)), pseudo_labels] #? confidence level

            # Add the data and pseudo labels to the heap, along with the corresponding logits
            for data, p_label, logit in zip(x, pseudo_labels, logits):
                heapq.heappush(top_data, (logit.item(), (data, p_label))) #? Save logit, (data, label) into a list

                # If the heap size exceeds 1024, remove the smallest element
                if len(top_data) > self.configs.alg_hparams["memory"]:
                    heapq.heappop(top_data) #? Remove the element with lowest logit

        # Extract the data and pseudo labels, discarding the logit values
        top_data = [item[1] for item in top_data] #? Remove logit so that final list is (data, label)
        top_dataset = MemoryDataset(top_data)

        #* Combine new target domain into memory
        combined_dataset = torch.utils.data.ConcatDataset([self.M.dataset, top_dataset])
        self.M = torch.utils.data.DataLoader(combined_dataset, batch_size=self.configs.train_params["batch_size"], shuffle=True)


    def BuildFeatureBank(self, joint_loader): #? Initialise the feature bank with previously trained feature extractor & encoder
        print("Initialising feature bank")
        custom_sampler = CustomRandomSampler(joint_loader.dataset)
        dataloader_with_custom_sampler = torch.utils.data.DataLoader(joint_loader.dataset, batch_size=self.configs.train_params["batch_size"], sampler=custom_sampler)
        self.B = FeatureBankDataset(dataloader_with_custom_sampler, self.g, self.feature_extractor, self.configs)
        self.B_loader = torch.utils.data.DataLoader(self.B, batch_size=self.configs.alg_hparams['n_negative'], shuffle=True)

    def UpdateKey(self, x, idx):
        new_k = self.g(self.feature_extractor(x))
        updated_k = self.configs.alg_hparams['momentum']*self.B.k[idx] -(1-self.configs.alg_hparams['momentum'])*new_k
        self.B.update_key(updated_k, idx)

    def optimise(self, g_t, g_s, g_dm):
        self.G = torch.cat([torch.neg(g_s).unsqueeze(0), torch.neg(g_dm).unsqueeze(0)], dim=0)
        self.g_t = g_t
        self.v_optimizer.step(self.closure)
        return self.v.detach()

    def closure(self):
        self.v_optimizer.zero_grad()
        loss =  self.objective(self.v)
        loss.backward()
        return loss

    def objective(self, v):
        u = torch.exp(v)
        term1 = 0.5 * u.t() @ self.G @ self.G.t() @ u
        term2 = self.g_t @ self.G.t() @ u
        return term1 - term2

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

class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, top_data):
        self.data = top_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

class FeatureBankDataset(torch.utils.data.Dataset):
    def __init__(self, dataloader, g, feature_extractor, config):
        self.data = dataloader.dataset
        
        # Initialize tensor to store transformed values
        self.k = torch.zeros((len(self.data), config.alg_hparams['len_encoded']))
        
        with torch.no_grad():
            # Iterate through the dataset and apply the transformation
            for idx, (x, _) in tqdm(enumerate(self.data), total=len(self.data)):
                x = x.unsqueeze(0)
                x = x.to(config.device)
                self.k[idx] = g(feature_extractor((x)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, = self.data[idx]  # Corrected unpacking
        k = self.k[idx]

        return sample, k, idx  # Return both data and index

    def update_key(self, new_value, idx):
        self.k[idx] = new_value

class CustomRandomSampler(torch.utils.data.RandomSampler):
    def __iter__(self):
        self.indices = list(super().__iter__())
        return iter(self.indices)