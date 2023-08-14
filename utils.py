import random
import numpy as np
import torch

def accuracy(pred, y):
    correct = (pred == y).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class KMeans:
    def __init__(self, n_clusters, n_features, device='cpu'):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.device = device
        self.centroids = torch.randn(n_clusters, n_features, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.centroids], lr=0.01)

    def __call__(self, x):
        # Compute pairwise distances between data and centroids
        distances = torch.cdist(x, self.centroids)
        
        # Assign each data point to the closest centroid
        cluster_assignments = torch.argmin(distances, dim=1)
        
        return cluster_assignments

    def optimize(self, dataloader, epochs=40):
        for epoch in range(epochs):
            cluster_counts = torch.zeros(self.n_clusters, device=self.device)

            # Optimise centroids
            for x in dataloader:
                x = x.to(self.device)

                # Reset the gradients
                self.optimizer.zero_grad()

                # Compute pairwise distances
                distances = torch.cdist(x, self.centroids)

                # Compute loss
                loss = torch.min(distances, dim=1)[0].mean()

                # Compute gradients
                loss.backward()

                # Update centroids
                self.optimizer.step()

                # Update cluster counts
                cluster_assignments = self(x)
                cluster_counts += torch.bincount(cluster_assignments, minlength=self.n_clusters)

            # Handle empty clusters
            with torch.no_grad():
                for i in range(self.n_clusters):
                    if cluster_counts[i]:
                        continue
                    # Select a non-empty cluster randomly
                    non_empty_clusters = (cluster_counts != 0).nonzero().squeeze()
                    non_empty_cluster = non_empty_clusters[torch.randint(0, len(non_empty_clusters), (1,))]

                    # Use its centroid with a small random perturbation as the new centroid for the empty cluster
                    self.centroids[i] = self.centroids[non_empty_cluster] + 0.01 * torch.randn_like(self.centroids[non_empty_cluster])

def combine_dataloaders(*dataloaders, batch_size = 64):
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([dataloader.dataset for dataloader in dataloaders])
    
    # Create a new DataLoader from the combined dataset
    combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    return combined_dataloader