from collections import defaultdict
import copy
from itertools import cycle
import numpy as np
import os
import torch
from torch.distributions import Normal
from algorithms.BaseAlgo import BaseAlgo
from architecture.Discriminator import Discriminator, ReverseLayerF

class FRIDA(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.G = GAN()
        self.D = Discriminator_GAN()
        self.discriminator = Discriminator(configs)
        self.taskloss = torch.nn.CrossEntropyLoss()
        self.feature_extractor_optimiser = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        self.classifier_optimiser = torch.optim.Adam(
                self.feature_extractor.parameters(),
                lr=configs.lr,
                weight_decay=configs.weight_decay
            )
        self.discriminator_optimiser = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay
        )
        self.fe_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.feature_extractor_optimiser,
                                      step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.classifier_optimiser,
                                              step_size=configs.step_size, gamma=configs.gamma)
        self.hparams = configs

    def epoch_train(self, src_loader, trg_loader, epoch, device):
        # Sending models to GPU
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.discriminator.to(device)

        # Make models to be train
        self.feature_extractor.train()
        self.classifier.train()
        self.discriminator.train()

        combined_loader = zip(cycle(src_loader), trg_loader)
        loss_dict = defaultdict(float)
        for step, (source, target) in enumerate(combined_loader):
            src_x, src_y, trg_x = source[0], source[1], target[0]
            src_x, src_y, trg_x = src_x.to(device), src_y.to(device), trg_x.to(device)

            #* Generate synthetic data
            z = torch.randn(trg_x.size(0), 512).to(device)
            with torch.no_grad():
                synthetic_x = self.G(z, src_y).to(device).to(device)

            #* Zero grads
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()
            self.discriminator_optimiser.zero_grad()

            p = float(step + epoch * len(trg_loader)) / self.hparams.n_epoch / len(trg_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            synthetic_domain_labels = torch.zeros(len(synthetic_x)).long().cuda()
            trg_domain_labels = torch.ones(len(trg_x)).long().cuda()

            #* Forward pass
            src_feature = self.feature_extractor(src_x)
            src_output = self.classifier(src_feature)
            synthetic_feature = self.feature_extractor(synthetic_x)
            synthetic_pred = self.classifier(synthetic_feature)
            trg_feat = self.feature_extractor(trg_x)

            #* Task classification
            src_cls_loss = self.taskloss(src_output.squeeze(), src_y)
            synthetic_cls_loss = self.taskloss(synthetic_pred.squeeze(), src_y)

            #* Domain classification
            # Source
            synthetic_feat_reversed = ReverseLayerF.apply(synthetic_feature, alpha)
            synthetic_domain_pred = self.discriminator(synthetic_feat_reversed)
            synthethic_domain_loss = self.taskloss(synthetic_domain_pred, synthetic_domain_labels)
            # Target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.discriminator(trg_feat_reversed)
            trg_domain_loss = self.taskloss(trg_domain_pred, trg_domain_labels)

            domain_loss = synthethic_domain_loss + trg_domain_loss

            #* Information bottleneck (IB) loss
            combined_feat = torch.cat((src_feature, trg_feat), dim = 0)
            ib_loss = information_bottleneck_loss(combined_feat)

            #* Backward pass
            loss = self.hparams.src_wt * src_cls_loss + self.hparams.src_wt * synthetic_cls_loss + self.hparams.da_wt * domain_loss + self.hparams.ib_wt * ib_loss
            loss.backward()
            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()
            self.discriminator_optimiser.step()

            #* Log the losses
            loss_dict["avg_loss"] += loss.item() / len(trg_x)
            loss_dict["avg_src_loss"] += src_cls_loss.item() / len(src_x)
            loss_dict["avg_domain_loss"] += domain_loss.item() / len(trg_x)
            loss_dict["avg_ib_loss"] += ib_loss.item() / len(trg_x)

        #* Adjust learning rate
        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        #* Train the GAN
        if epoch == self.configs.n_epoch-1:
            # Combine current samples with previous synthetic data to be used to train GAN
            confident_target_set = get_confident_labels(trg_loader, self.feature_extractor, self.classifier, device)
            synthetic_set = create_synthetic_set(self.G, self.hparams.num_class, 20000, device)
            combined_dataset = torch.utils.data.ConcatDataset([confident_target_set, synthetic_set])
            combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=256, shuffle=True)

            # Update GAN with combined dataloader
            self.train_GAN(combined_dataloader, device)

        return loss_dict


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

        #* Train GAN
        # initialise gan & discriminator network
        for data in train_loader:
            x, y = data[0], data[1]
            x, y = x.to(device), y.to(device)
            self.G.set_dims(x, y)
            self.D.set_dims(x, y)
            self.G_optimiser, self.D_optimiser = torch.optim.Adam(self.G.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay),torch.optim.Adam(self.D.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)
            break 
        # train gan & discriminator
        self.train_GAN(train_loader, device)

    def train_GAN (self, loader, device, n_epoch = 40):
        self.G.to(device)
        self.D.to(device)
        self.G.train()
        self.D.train()

        for epoch in range(n_epoch):
            for step, data in enumerate(loader):
                p = float(step + epoch * len(loader)) / n_epoch / len(loader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                self.G_optimiser.zero_grad()
                self.D_optimiser.zero_grad()

                real_x, y = data[0], data[1]
                real_x, y = real_x.to(device), y.to(device)

                real_domain_label = torch.zeros(len(real_x)).long().cuda()
                fake_domain_labels = torch.ones(len(real_x)).long().cuda()

                # Generate synthetic samples
                z = torch.randn(real_x.size(0), 512).to(device)
                fake_x = self.G(z, y)

                # Get discriminator prediction
                # real samples
                real_x_reversed = ReverseLayerF.apply(real_x, alpha)
                real_domain_pred = self.D(real_x_reversed)
                real_domain_loss = self.taskloss(real_domain_pred, real_domain_label)
                # fake samples
                fake_x_reversed = ReverseLayerF.apply(fake_x, alpha)
                fake_domain_pred = self.D(fake_x_reversed)
                fake_domain_loss = self.taskloss(fake_domain_pred, fake_domain_labels)

                loss = real_domain_loss + fake_domain_loss
                loss.backward()
                self.G_optimiser.step()
                self.D_optimiser.step()

class GAN(torch.nn.Module):
    def __init__(self, Z_dim = 512, h_dim=100):
        super().__init__()
        self.Z_dim = Z_dim
        self.h_dim = h_dim
        self.X_shape = None
        self.y_dim = None
        self.fc1 = None
        self.fc2 = None
        self.sigmoid = torch.nn.Sigmoid()
    
    def set_dims(self, x, y):
        self.X_shape = x.size()[1:]
        self.y_dim = 1
        self.fc1 = torch.nn.Linear(self.Z_dim + self.y_dim, self.h_dim)
        self.fc2 = torch.nn.Linear(self.h_dim, self.X_shape[0] * self.X_shape[1])
    
    def forward(self, z, y):
        if self.fc1 is None or self.fc2 is None:
            raise ValueError("Dimensions not set. Please call set_dims() first.")
        x = torch.cat([z, y.unsqueeze(1)], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.X_shape[0], self.X_shape[1])
        return x

class Discriminator_GAN(torch.nn.Module):
    """Discriminator model for GAN."""
    def __init__(self):
        super().__init__()
        self.X_shape = None
        self.y_dim = None

    def set_dims(self, x, y):
        self.X_shape = x.size()[1:]
        self.y_dim = 1
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(self.X_shape[0] * self.X_shape[1], 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(True),
            torch.nn.Linear(100, 2),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """Forward the discriminator."""
        x = x.view(-1, self.X_shape[0] * self.X_shape[1])
        x = self.layer(x)
        return x

def information_bottleneck_loss(encoded_features):
    prior = Normal(torch.zeros_like(encoded_features), torch.ones_like(encoded_features))
    posterior = Normal(encoded_features, torch.ones_like(encoded_features))  # Assuming unit variance for simplicity
    kl_divergence = torch.distributions.kl_divergence(posterior, prior)
    info_loss = torch.mean(kl_divergence)
    return info_loss

def get_confident_labels (dataloader, fe, classifier, device, rho = 0.9):
    # Set the model to evaluation mode
    fe.eval()
    classifier.eval()

    all_results = []
    with torch.no_grad(): 
        for batch in dataloader:
            inputs = batch[0].to(device)
            logits = classifier(fe(inputs))
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)

            for i in range(inputs.size(0)):
                confidence = max_probs[i].item()  # Get the confidence level as a Python number
                if confidence > rho:
                    sample = inputs[i].tolist()  # Convert sample tensor to list
                    prediction = preds[i].item()  # Get the prediction as a Python number
                    all_results.append((sample, prediction))

    # Separate features, labels, and confidences
    x = [item[0] for item in all_results]
    preds = [item[1] for item in all_results]
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).to(device)
    pred_tensor = torch.LongTensor(preds).to(device)

    pseudolabel_dataset = torch.utils.data.TensorDataset(x_tensor, pred_tensor)

    return pseudolabel_dataset

def create_synthetic_set (G, num_class, N, device):
    G.to(device)
    G.eval()
    z = torch.randn(N, 512).to(device)
    y = torch.randint(low=0, high=num_class, size=(N,)).to(device)
    with torch.no_grad():
        synthetic_samples = G(z, y)

    synthetic_dataset = torch.utils.data.TensorDataset(synthetic_samples, y)
    return synthetic_dataset