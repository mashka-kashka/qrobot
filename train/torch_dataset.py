import torch.utils.data


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x_val = self.X.iloc[idx].values
        y_val = self.y.iloc[idx]
        return x_val, y_val

    def __len__(self):
        return len(self.y)