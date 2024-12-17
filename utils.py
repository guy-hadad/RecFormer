import json
import torch
import torch.nn as nn

MAX_VAL = 1e4

def read_json(path, as_int=False):
    with open(path, 'r') as f:
        raw = json.load(f)
        if as_int:
            data = dict((int(key), value) for (key, value) in raw.items())
        else:
            data = dict((key, value) for (key, value) in raw.items())
        del raw
        return data



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

MAX_VAL = 1e9

class Ranker(nn.Module):
    def __init__(self, metrics_ks, num_negatives=29, seed=None):
        """
        Args:
            metrics_ks (list): List of k values for metrics (e.g., [1, 5, 10]).
            num_negatives (int): Number of negative samples per target item.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__()
        self.ks = metrics_ks
        self.num_negatives = num_negatives
        self.ce = nn.CrossEntropyLoss()

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

    def forward(self, scores, labels):
        """
        Args:
            scores (torch.Tensor): [batch_size, num_classes]
            labels (torch.Tensor): [batch_size, num_labels]

        Returns:
            List[float]: Metrics in the order of:
            [recall@k, ndcg@k for each k in ks, ..., MRR, loss]
            
            Specifically:
            For each k in self.ks, we append recall@k, ndcg@k.
            After that, we append MRR and finally the loss.
        """
        batch_size, num_classes = scores.size()
        _, num_labels = labels.size()

        sampled_scores_list = []
        targets = []  # Will hold the index of the true item within the candidate set (always 0)
        
        for i in range(batch_size):
            seq = labels[i].tolist()
            print("seq", seq)
            true_item = seq[-1]  # The last item is the next item to predict
            print("true_item", true_item)
            # Exclude all items in the sequence from negative sampling
            negatives = self.sample_negatives(num_classes, seq, self.num_negatives)
            print("negatives", negatives)
            # Construct candidate set: true_item + negatives
            candidate_items = [true_item] + negatives

            print("candidate_items", candidate_items)
            # Extract scores for these candidates
            candidate_scores = scores[i, candidate_items]
            print("candidate_scores", candidate_scores)

            # candidate_scores shape: [30] (1 + num_negatives)
            sampled_scores_list.append(candidate_scores)
            targets.append(0)  # The true item is always at index 0

        if len(sampled_scores_list) == 0:
            # No labels found, return zeros
            num_metrics = (2 * len(self.ks)) + 1  # recall@k and ndcg@k for each k, plus MRR
            return [0.0] * num_metrics + [0.0]

        # Convert lists to tensors
        sampled_scores = torch.stack(sampled_scores_list, dim=0)  # [batch_size * num_labels, 1+num_negatives]
        targets_tensor = torch.tensor(targets, device=scores.device, dtype=torch.long)

        # Compute loss
        loss = self.ce(sampled_scores, targets_tensor).item()

        # Rank candidates and compute metrics
        # Sort in descending order
        sorted_scores, sorted_indices = torch.sort(sampled_scores, descending=True, dim=1)

        # Find the rank of the true item (which is index 0 in candidate_items)
        # sorted_indices gives the rearranged indices of candidates.
        # We look for where '0' (the true label index within the candidate set) appears.
        true_positions = (sorted_indices == 0).nonzero(as_tuple=False)
        # shape: [batch_size, 2], each row: [row_idx, col_idx], col_idx is the position in sorted list
        ranks = true_positions[:, 1].float()  # zero-based rank

        # Compute metrics
        res = []
        for k in self.ks:
            # recall@k: 1 if rank < k else 0
            recall_k = (ranks < k).float().mean().item()
            res.append(recall_k)

            # ndcg@k: (1/log2(rank+2)) if rank < k else 0
            ndcg_k = ((ranks < k).float() * (1.0 / torch.log2(ranks + 2.0))).mean().item()
            res.append(ndcg_k)

        # MRR: mean(1/(rank+1))
        mrr = (1.0 / (ranks + 1.0)).mean().item()
        res.append(mrr)

        # Append loss
        res.append(loss)
        return res

    def sample_negatives(self, num_classes, true_items, num_negatives):
        """
        Samples negative labels excluding the given true_items.
        
        Args:
            num_classes (int): Total number of classes.
            true_items (list): Items to exclude.
            num_negatives (int): Number of negatives.
        
        Returns:
            List[int]: Indices of negative samples.
        """
        all_items = set(range(num_classes))
        excluded = set(true_items)
        candidates = list(all_items - excluded)

        if len(candidates) < num_negatives:
            raise ValueError(f"Not enough candidates for negative sampling: Required {num_negatives}, got {len(candidates)}")
        
        return random.sample(candidates, num_negatives)
