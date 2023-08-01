import torch
import numpy as np

def weighted_f_score(
    outputs: torch.IntTensor,
    targets: torch.IntTensor,
    weights: torch.FloatTensor,
) -> float:
    """
    Weighted F-score as implemented in https://ndownloader.figstatic.com/files/7128245
    TODO: add thresh as argument
    """
    
    with torch.no_grad():

        #print(outputs, targets, weights)
        
        # Get weighted true positives, total positives and total true
        w_true_pos = (weights * torch.logical_and(targets == outputs, targets == 1)).sum(dim=1)
        w_total_pos = (weights * (outputs == 1)).sum(dim=1)
        w_total_true = (weights * (targets == 1)).sum(dim=1)

        # Get precision and recall for each output
        n_one_more_than_thresh = (outputs.sum(dim=1) > 0).sum()
        n_samples = outputs.shape[0]
        precision = torch.nan_to_num(torch.nan_to_num(w_true_pos/w_total_pos).sum(dim=0)/n_one_more_than_thresh)
        recall = torch.nan_to_num(torch.nan_to_num(w_true_pos/w_total_true).sum(dim=0)/n_samples)        

        # Get weighted F-score
        f_score = torch.nan_to_num(2 * precision * recall / (precision + recall))

        return f_score.item()
    
    
def weighted_f_score_by_group(
    outputs: torch.IntTensor,
    targets: torch.IntTensor,
    weights: torch.FloatTensor,
    groups: list[int | str]
) -> float:
    """
    Weighted F-score as implemented in https://ndownloader.figstatic.com/files/7128245
    TODO: add thresh as argument
    """
    
    with torch.no_grad():
    
        # Initialize total F-score & unique subontologies
        unique_groups = np.unique(groups)
        f_scores = torch.zeros(len(unique_groups))

        # Get F-score by subontology
        for i, group in enumerate(unique_groups):
            group_mask = np.array(groups) == group
            f_scores[i] = weighted_f_score(targets[:, group_mask], 
                                           outputs[:, group_mask], 
                                           weights[group_mask])
        return f_scores.mean().item()