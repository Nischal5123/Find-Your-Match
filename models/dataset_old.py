import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    )
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
dataset_path = 'datasets/'  # REPLACE THIS WITH OUR DATASET "TUPLE-1", "TUPLE-2" , "SCORE"

class PairDataset(Dataset):
    def __init__(self, path, max_length) -> None:
        super(PairDataset).__init__()

        original_data = pd.read_csv(path,index_col='split')

        self.data=original_data[original_data.columns[:2]]
        self.labels=original_data['relevant']
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def tokenize(self,word):
        #implement a tokenizer placeholder
        return np.random.uniform(1,self.max_length)


    def __getitem__(self, index):
        local_tuple = self.data.local_tuple[index].split()
        external_tuple = self.data.external_tuple[index].split()
        label=self.labels[index]
    
        local_tuple = np.array([self.tokenize(x) for x in local_tuple])
        external_tuple = np.array([self.tokenize(x) for x in external_tuple])
       


        local_tuple = np.pad(local_tuple, (0, self.max_length - local_tuple.shape[0]), 'constant', constant_values=(0, 0))
        external_tuple = np.pad(external_tuple, (0, self.max_length - external_tuple.shape[0]), 'constant', constant_values=(0, 0))
       

        return local_tuple, external_tuple,label


if __name__ == "__main__":
    maxlength=500
    batch_size=50
    dataset = PairDataset(dataset_path+"positive.csv", maxlength)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,  collate_fn=None)

    for batch_idx, batch in enumerate(dataloader):
        sent1,sent2, target = batch
        print(sent1.shape, target.shape)
