import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.model_selection import train_test_split
import helper
from transformers import GPT2Config, GPT2Model,GPT2Tokenizer


class TupleData(Dataset):
    def __init__(self, dataset, languageModel='gpt2'):
        self.tokenizer = None
        self.pad_token = None

        if languageModel == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.local_tuple = list(dataset["local_tuple"])
        self.external_tuple = list(dataset["external_tuple"])
        self.relevant = list(dataset["relevant"])

    def __len__(self):
        return len(self.relevant)

    def __getitem__(self, idx):
        return self.local_tuple[idx], self.external_tuple[idx], self.relevant[idx]





class GPT2_WordEncoder(torch.nn.Module):
    def __init__(self, tokenizer, gpt_model_name='gpt2'):
        super().__init__()
        self.model = GPT2Model.from_pretrained(gpt_model_name)
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, inputs):
        embeddings = self.model(**inputs)
        return embeddings.last_hidden_state


class Attention(nn.Module):
    def __init__(self, hidden_dim, output_dim=1024, attn_dim=512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attn_dim = attn_dim

        self.w1 = nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.tanh = nn.Tanh()
        self.w2 = nn.Linear(self.output_dim, self.attn_dim)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(self.attn_dim, 1)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, encoder_outputs):
        src_len = encoder_outputs.shape[1]

        output_fw = encoder_outputs[:, :, 0:self.hidden_dim]
        output_bw = encoder_outputs[:, :, self.hidden_dim:]

        hidden_states = torch.cat((output_fw, output_bw), -1)

        # Obtaining the attention weights
        weighted_states = self.w1(hidden_states)
        activated_states = self.tanh(weighted_states)
        score_weights = self.w2(activated_states)
        attention_weights = self.softmax(score_weights)

        # Applying attention to the matrix with hidden states
        attentional_vector = torch.bmm(torch.transpose(attention_weights, 2, 1), hidden_states)
        attentional_vector = self.fc(torch.transpose(attentional_vector, 2, 1)).squeeze(2)
        attentional_vector = self.leakyRelu(attentional_vector)

        return attentional_vector


class AttentionBiGRU(torch.nn.Module):
    def __init__(self, tokenizer, embed_dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = GPT2_WordEncoder(tokenizer)
        self.gru = nn.GRU(embed_dim, hidden_dim // 2, bidirectional=True)
        self.attention = Attention(hidden_dim // 2)
        self.apply(self._init_weights)
        self._eps = 10e-8  # to avoid division by 0

    def _init_weights(self, module):
        if isinstance(module, nn.GRU):
            for param_name, weights in module.named_parameters():
                if "weight_hh" in param_name:
                    torch.nn.init.eye_(weights)
                if "weight_ih" in param_name:
                    torch.nn.init.orthogonal_(weights)
                if "bias" in param_name:
                    torch.nn.init.constant_(weights, 0.5)

    def forward(self, x1, x2):
        local = self.embed(x1)
        external = self.embed(x2)

        out_l, ht_l = self.gru(local)
        out_e, ht_e = self.gru(external)

        attn_l = self.attention(out_l)
        attn_e = self.attention(out_e)

        distance = F.cosine_similarity(attn_l, attn_e, dim=-1, eps=self._eps)

        return distance



def pad_collate(batch, tokenizer):
  (xx1, xx2, yy) = zip(*batch)

  xx1_inputs = tokenizer(xx1, return_tensors="pt", padding=True)
  xx2_inputs = tokenizer(xx2, return_tensors="pt", padding=True)

  yy = torch.tensor(yy)

  return xx1_inputs, xx2_inputs, yy

def main():
    googleTrainData, googleTestData = helper.createDatasets("datasets/google")
    googleTrain = TupleData(googleTrainData, 'gpt2')
    googleTest = TupleData(googleTestData, 'gpt2')

    B = 256
    # B=2
    train_loader = DataLoader(googleTrain, batch_size=B, shuffle=True,
                              collate_fn=lambda b: pad_collate(b, googleTrain.tokenizer))
    test_loader = DataLoader(googleTest, batch_size=200, shuffle=False,
                             collate_fn=lambda b: pad_collate(b, googleTest.tokenizer))
    for batch_idx, batch in enumerate(train_loader):
         = batch
        print(len(xx1_input_pad), len(x1_lens))



if __name__== "__main__":
    main()