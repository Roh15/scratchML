import numpy as np


class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=True, seed=42):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Array shape mismatch: X.shape[0] = {X.shape[0]} != y.shape[0] = {y.shape[0]}")
        np.random.seed(seed)
        if shuffle:
            shuffled_indices = np.random.permutation(X.shape[0])
            self.X = X[shuffled_indices]
            self.y = y[shuffled_indices]
        else:
            self.X = X
            self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]

    def __iter__(self):
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            yield self.X[start_idx:end_idx], self.y[start_idx:end_idx]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
