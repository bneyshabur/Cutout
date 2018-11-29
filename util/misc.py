import numpy as np
from torch.utils.data import SubsetRandomSampler

def get_sampler_classifier(dataloader, seed, proportion_training):

    if proportion_training == 1:
        return None

    labels = np.array(dataloader.dataset.train_labels)
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    num_training_samples_per_label = int(proportion_training * len(labels) / n_labels)
    training_subset_idx = []
    for label in unique_labels:
        label_set = np.where(labels == label)[0]
        label_set_len = len(label_set)
        # Subset indices for training
        np.random.seed(seed)
        label_training_subset = np.random.choice(label_set, size=num_training_samples_per_label, replace=False)
        training_subset_idx += list(label_training_subset)
    return SubsetRandomSampler(training_subset_idx)
