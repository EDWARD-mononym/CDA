from itertools import cycle
import os
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from algorithms.BaseAlgo import BaseAlgo
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class NRC(BaseAlgo):
    def __init__(self, configs) -> None:
        super().__init__(configs)


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

        self.fe_lr_scheduler = StepLR(self.feature_extractor_optimiser,
                                      step_size=configs.step_size, gamma=configs.gamma)
        self.classifier_lr_scheduler = StepLR(self.classifier_optimiser,
                                              step_size=configs.step_size, gamma=configs.gamma)
        self.configs = configs
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

                #* Zero grads
                self.feature_extractor_optimiser.zero_grad()
                self.classifier_optimiser.zero_grad()

                #* Forward pass
                pred = self.classifier(self.feature_extractor(x))

                #* Loss
                loss = self.cross_entropy_label_smooth(pred, y,self.configs.num_class, device, epsilon=0.1)
                loss.backward()

                #* Step
                self.feature_extractor_optimiser.step()
                self.classifier_optimiser.step()

                running_loss += loss.item()

            #* Adjust learning rate
            self.fe_lr_scheduler.step()
            self.classifier_lr_scheduler.step()

            #* Save best model
            # Print average loss every 'print_every' steps
            if (epoch + 1) % self.configs.print_every == 0:
                avg_loss = running_loss / len(train_loader)
                print(f"Average Loss: {avg_loss:.4f}")
            print("-" * 30)  # Print a separator for clarity

            # * Save best model
            epoch_acc = evaluator.test_domain(self, test_loader)
            if epoch_acc > best_acc:
                torch.save(self.feature_extractor.state_dict(), os.path.join(save_path, f"{source_name}_feature.pt"))
                torch.save(self.classifier.state_dict(), os.path.join(save_path, f"{source_name}_classifier.pt"))
    def epoch_train(self, src_loader, trg_loader, epoch, device):
        # Freeze the classifier
        self.device = device
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        # send to device
        self.feature_extractor.to(device)
        self.classifier.to(device)
        self.feature_extractor.train()

        combined_loader = zip(cycle(src_loader), trg_loader)

        loss_dict = defaultdict(float)


        fea_bank, score_bank = self.build_feat_score_bank(trg_loader)

        for step, (trg_x, _, trg_idx) in enumerate(trg_loader):
            trg_x, trg_idx = trg_x.to(device), trg_idx.to(device)

            #* Zero grads
            self.feature_extractor_optimiser.zero_grad()
            self.classifier_optimiser.zero_grad()

            #* Forward pass
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)
            softmax_out = nn.Softmax(-1)(trg_pred)

            with torch.no_grad():
                output_f_norm = F.normalize(trg_feat)

                # Directly store the normalized output on the GPU
                fea_bank[trg_idx] = output_f_norm.detach()
                score_bank[trg_idx] = softmax_out.detach()

                # Calculate distance without moving to CPU
                distance = output_f_norm @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=5 + 1)
                idx_near = idx_near[:, 1:]

                score_near = score_bank[idx_near]
                fea_near = fea_bank[idx_near]

                fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)
                distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))
                _, idx_near_near = torch.topk(distance_, dim=-1, largest=True, k=5 + 1)

                idx_near_near = idx_near_near[:, :, 1:]
                trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)

                match = (idx_near_near == trg_idx_).sum(-1).float()
                weight = torch.where(match > 0., match, torch.ones_like(match).fill_(0.1))

                weight_kk = weight.unsqueeze(-1).expand_as(idx_near_near).fill_(0.1)
                score_near_kk = score_bank[idx_near_near]

                weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)
                score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, self.configs.num_class)

                score_self = score_bank[trg_idx]


            # start gradients
            output_re = softmax_out.unsqueeze(1).expand(-1, 5 * 5,
                                                        -1)  # batch x C x 1
            const = torch.mean(
                (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                 weight_kk.cuda()).sum(
                    1))  # kl_div here equals to dot product since we do not use log for score_near_kk
            loss = torch.mean(const)

            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

            loss += torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))


            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + self.configs.epsilon))
            loss += gentropy_loss


            #* Compute loss
            loss.backward()

            #* update weights
            self.feature_extractor_optimiser.step()
            self.classifier_optimiser.step()

            # save average losses
            loss_dict["avg_loss"] += loss.item() / len(trg_x)
            loss_dict["avg_ent_loss"] += gentropy_loss.item() / len(trg_x)


        #* Adjust learning rate
        self.fe_lr_scheduler.step()
        self.classifier_lr_scheduler.step()

        return loss_dict


    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)

                features = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)

        preds = torch.cat((preds))
        feas = torch.cat((feas))

        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)
        return pred_label

    def cross_entropy_label_smooth(self, inputs, targets, num_classes, device, epsilon=0.1):
        logsoftmax = nn.LogSoftmax(dim=1)

        log_probs = logsoftmax(inputs)

        targets = torch.zeros(log_probs.size()).to(device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - epsilon) * targets + epsilon / num_classes

        loss = (- targets * log_probs).mean(0).sum()

        return loss

    def EntropyLoss(self, input_):
        mask = input_.ge(0.0000001)
        mask_out = torch.masked_select(input_, mask)
        entropy = - (torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0))

    def build_feat_score_bank(self, data_loader):
        # num_samples = len(data_loader.dataset)
        fea_bank = torch.empty(0).cuda()
        score_bank = torch.empty(0).cuda()

        self.feature_extractor.eval()
        self.classifier.eval()

        # Process data batch by batch
        for data in data_loader:
            batch_data = data[0].cuda()  # Assuming the first element in your batch is the data. Adjust as needed.

            batch_feat = self.feature_extractor(batch_data)
            norm_feats = F.normalize(batch_feat)
            batch_pred = self.classifier(batch_feat)
            batch_probs = nn.Softmax(dim=-1)(batch_pred)


            # Update the banks
            fea_bank = torch.cat((fea_bank, norm_feats.detach()), 0)
            score_bank = torch.cat((score_bank, batch_probs.detach()), 0)
        #
        # # create
        # full_data = data_loader.dataset.x.cuda()
        # # full_data = full_data.to(self.device)
        #
        # full_feat = self.feature_extractor(full_data)
        # norm_feats = F.normalize(full_feat)
        # full_pred = self.classifier(full_feat)
        # full_probs = nn.Softmax(-1)(full_pred)
        #
        # fea_bank = norm_feats.detach()
        # score_bank = full_probs.detach()  #

        return fea_bank, score_bank


# with torch.no_grad():
#     output_f_norm = F.normalize(trg_feat)
#     output_f_ = output_f_norm.cpu().detach().clone()
#
#     fea_bank[trg_idx] = output_f_.detach().clone().cpu()
#     score_bank[trg_idx] = softmax_out.detach().clone()
#
#     distance = output_f_ @ fea_bank.T
#     _, idx_near = torch.topk(distance,
#                              dim=-1,
#                              largest=True,
#                              k=5 + 1)
#     idx_near = idx_near[:, 1:]  # batch x K
#     score_near = score_bank[idx_near]  # batch x K x C
#
#     fea_near = fea_bank[idx_near]  # batch x K x num_dim
#     fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
#     distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
#     _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
#                                   k=5 + 1)  # M near neighbors for each of above K ones
#     idx_near_near = idx_near_near[:, :, 1:].to(device)  # batch x K x M
#     trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)
#     match = (idx_near_near == trg_idx_).sum(-1).float()  # batch x K
#     weight = torch.where(match > 0., match, torch.ones_like(match).fill_(0.1))  # batch x K
#
#     weight_kk = weight.unsqueeze(-1).expand(-1, -1,
#                                             5)  # batch x K x M
#     weight_kk = weight_kk.fill_(0.1)
#
#     # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
#     # weight_kk[idx_near_near == trg_idx_]=0
#
#     score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
#     # print(weight_kk.shape)
#     weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
#                                             -1)  # batch x KM
#
#     score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
#                                                     self.configs.num_class)  # batch x KM x C
#
#     score_self = score_bank[trg_idx]
