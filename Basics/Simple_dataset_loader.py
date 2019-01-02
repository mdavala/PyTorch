import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ExampleDataset(Dataset):
    """Example Dataset"""

    def __init__(self, csv_file):
        """ 
        csv_file (string): Path to the csv file containing data.
        """
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.data_frame[idx]

# get dataset object 
example_dataset = ExampleDataset('my_data_file.csv')

# batch size: number of samples returned per iteration
# shuffle: Flag to shuffle the data before reading
# num_workers: used for paralle processing the data
example_data_loader = DataLoader(example_dataset, batch_size=4, shuffle=True, num_workers=4)

#print batch index and batch for each batch
for batch_index, batch in enumerate(example_data_loader):
    print(batch_index, batch)