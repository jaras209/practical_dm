import torch
from dataset_torch import DialogDataset, DialogDataLoader
from pathlib import Path

if __name__ == '__main__':
    train = DialogDataset(dataset_type='train', k=10, domains=['restaurant', 'hotel'])
    # val = DialogDataset(dataset_type='val', k=10)
    # test = DialogDataset(dataset_type='test', k=10)

    train_loader = DialogDataLoader(train, batch_size=3, batch_first=True)

    i = 0
    for batch in train_loader:
        print(batch)
        print(train_loader.to_string(batch))
        i += 1
        if i >= 3:
            break


