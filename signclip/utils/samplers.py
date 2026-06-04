import random
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    """
    Yields batches of size P * K, containing exactly P classes and K instances per class.
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.n_classes = n_classes  # P
        self.n_samples = n_samples  # K
        self.batch_size = self.n_classes * self.n_samples
        
        # Group dataset indices by their class label
        self.label_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            # Assuming dataset[idx] returns a dict with a 'label' key
            label = dataset[idx]['label']
            self.label_to_indices[label].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        
        # Calculate how many batches we can form per epoch
        # (This is an approximation to ensure we use most of the data)
        self.num_batches = len(dataset) // self.batch_size

    def __iter__(self):
        # Create a fresh copy of indices to shuffle for this epoch
        indices_by_label = {
            label: indices[:] for label, indices in self.label_to_indices.items()
        }
        for indices in indices_by_label.values():
            random.shuffle(indices)

        for _ in range(self.num_batches):
            batch = []
            # 1. Randomly pick P classes for this batch
            classes_for_batch = random.sample(self.labels, self.n_classes)
            
            # 2. Pick K instances for each chosen class
            for label in classes_for_batch:
                # If we run out of instances for a class, recycle them (shuffle again)
                if len(indices_by_label[label]) < self.n_samples:
                    indices_by_label[label] = self.label_to_indices[label][:]
                    random.shuffle(indices_by_label[label])
                
                # Pop K instances
                for _ in range(self.n_samples):
                    batch.append(indices_by_label[label].pop())
            
            yield batch

    def __len__(self):
        return self.num_batches