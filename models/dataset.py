import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model
import helper
from torch.utils.data import DataLoader
import logging
import tqdm

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    )

class TupleData(Dataset):
    def __init__(self, dataset, languageModel='gpt2'):
        self.tokenizer = None
        self.pad_token = None

        if languageModel == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            #self.pad_token = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]

        else:
            self.tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/LinkBERT-large',model_max_length=3500)#maxlength in google dataset is around 3400
            self.pad_token = self.tokenizer.pad_token_id

        self.local_tuple = list(dataset["local_tuple"])
        self.external_tuple = list(dataset["external_tuple"])
        self.relevant = list(dataset["relevant"])

    def __len__(self):
        return len(self.relevant)

    def __getitem__(self, idx):
        return self.tokenizer(self.local_tuple[idx], return_tensors="pt"), self.tokenizer(self.external_tuple[idx],
                                                                                          return_tensors="pt"), \
               self.relevant[idx]

def pad_collate(batch):
  (xx1, xx2, yy) = zip(*batch)
  xx1_input = tuple(x['input_ids'].squeeze(0) for x in xx1)
  xx2_input = tuple(x['input_ids'].squeeze(0) for x in xx2)

  xx1_attnMask = tuple(x['attention_mask'].squeeze(0) for x in xx1)
  xx2_attnMask = tuple(x['attention_mask'].squeeze(0) for x in xx2)

  x1_lens = [len(x) for x in xx1_input]
  x2_lens = [len(x) for x in xx2_input]

  xx1_input_pad = pad_sequence(xx1_input, batch_first=True, padding_value=0)
  xx2_input_pad = pad_sequence(xx2_input, batch_first=True, padding_value=0)

  xx1_mask_pad = pad_sequence(xx1_attnMask, batch_first=True, padding_value=0)
  xx2_mask_pad = pad_sequence(xx2_attnMask, batch_first=True, padding_value=0)

  yy = torch.tensor(yy)

  return xx1_input_pad, xx1_mask_pad, xx2_input_pad, xx2_mask_pad, yy, x1_lens, x2_lens


def main():
    googleTrainData, googleTestData = helper.createDatasets("datasets/google")
    googleTrain = TupleData(googleTrainData, 'gpt2')
    googleTest = TupleData(googleTestData, 'gpt2')

    B = 256
    train_loader = DataLoader(googleTrain, batch_size=B, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(googleTest, batch_size=200, shuffle=False, collate_fn=pad_collate)
    for batch_idx, batch in enumerate(train_loader):
        xx1_input_pad, xx1_mask_pad, xx2_input_pad, xx2_mask_pad, yy, x1_lens, x2_lens = batch
        print(len(xx1_input_pad), len(x1_lens))



if __name__== "__main__":
    main()