import torch
import torch.nn.functional as F

def InfoNCE_loss(queries, positives, tau=0.07):
    loss = 0
    
    for i, q in enumerate(queries):
        positive = positives[i]
        
        # Create the negatives by taking all other 1D tensors from queries except the current one
        negatives = torch.cat([queries[:i], queries[i+1:]])
        
        # Compute dot product between query and positive
        pos_similarity = q @ positive
        
        # Compute dot product between query and negatives
        neg_similarity = negatives @ q
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarity])
        
        # Apply temperature scaling
        logits /= tau
        
        # Compute the loss for the current 1D tensor
        loss += F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=queries.device)) #? cross_entropy = -log(softmax) = -log(e^{x}/SUM{e^{x_i}})
    
    return loss
