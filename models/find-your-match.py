import torch
torch.manual_seed(42)
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
import nlp
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

def createDatasets(folder="google", pos=True):
  df_p = pd.read_csv(folder+'/positive.csv')
  df_n = pd.read_csv(folder+'/negative.csv')
  df_merged = df_p #df_p.append(df_n, ignore_index=True)
  if not pos:
    df_merged = df_n
  y = df_merged['relevant']
  X = df_merged[['local_tuple', 'external_tuple']]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
  train = pd.concat([X_train, y_train], axis=1)
  test = pd.concat([X_test, y_test], axis=1)
  return train, test

googleTrainData, googleTestData = createDatasets("google")

#drugsTrainData, drugsTestData = createDatasets("drugs")
#summariesTrainData, summariesTestData = createDatasets("summaries") 

class TupleData(Dataset):
  def __init__(self, dataset, languageModel='gpt2'):

    self.tokenizer = None
    self.pad_token = None

    if languageModel == 'gpt2':
      self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
      self.tokenizer.padding_side = "left" 
      self.tokenizer.pad_token = self.tokenizer.eos_token
      #self.pad_token = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
    
    self.local_tuple = list(dataset["local_tuple"])
    self.external_tuple = list(dataset["external_tuple"])
    self.relevant = list(dataset["relevant"])
    max_local_seq_length = max(len(x) for x in dataset['local_tuple'])
    max_external_seq_length = max(len(x) for x in dataset['external_tuple'])
    self.max_seq_len = max(max_local_seq_length, max_external_seq_length)
  def __len__(self):
    return len(self.relevant)
  
  def __getitem__(self,idx):
    return self.local_tuple[idx], self.external_tuple[idx], self.relevant[idx]
    #return self.tokenizer.encode(self.local_tuple[idx], padding=True, max_length=1024, truncation=True, return_tensors="pt"), self.tokenizer.encode(self.external_tuple[idx],padding=True, max_length=1024, truncation=True, return_tensors="pt"), self.relevant[idx]
    #return self.tokenizer(self.tokenizer.bos_token + self.local_tuple[idx] + self.tokenizer.eos_token, max_length=1024, truncation=True, return_tensors="pt"), self.tokenizer(self.tokenizer.bos_token + self.external_tuple[idx] + self.tokenizer.eos_token, max_length=1024, truncation=True, return_tensors="pt"), self.relevant[idx]
  
googleTrain = TupleData(googleTrainData, 'gpt2')
googleTest = TupleData(googleTestData, 'gpt2')

def pad_collate(batch, tokenizer, max_length):
  (xx1, xx2, yy) = zip(*batch)

  xx1_inputs = tokenizer(xx1, return_tensors="pt", padding='max_length',max_length=512, truncation=True).to("cuda")#pad_sequence(xx1_input, batch_first=True, padding_value=pad_token)
  xx2_inputs = tokenizer(xx2, return_tensors="pt", padding='max_length', max_length=512, truncation=True).to("cuda")#pad_sequence(xx2_input, batch_first=True, padding_value=pad_token)
  
  yy = torch.tensor(yy)#.cuda()
  return xx1_inputs, xx2_inputs, yy

B=16
train_loader = DataLoader(googleTrain, batch_size=B, shuffle=True, collate_fn=lambda b: pad_collate(b, googleTrain.tokenizer, min(512, googleTrain.max_seq_len)))
test_loader = DataLoader(googleTest, batch_size=B, shuffle=True, collate_fn=lambda b: pad_collate(b, googleTest.tokenizer, min(512, googleTest.max_seq_len)))


class GPT2_WordEncoder(torch.nn.Module):
  def __init__(self, tokenizer, gpt_model_name='gpt2'):
    super().__init__()
    self.model = GPT2Model.from_pretrained(gpt_model_name)
    self.model.resize_token_embeddings(len(tokenizer))
    self.model.config.pad_token_id = self.model.config.eos_token_id
    for param in self.model.base_model.parameters():
      param.requires_grad = False
  def forward(self, inputs):
    embeddings = self.model(**inputs).last_hidden_state
    return embeddings
class SimpleModel(nn.Module):
  def __init__(self, tokenizer, embed_dim, hidden_dim=1024):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.embed = GPT2_WordEncoder(tokenizer)
    self.fc1 = nn.Linear(embed_dim*512, hidden_dim)
    self.text_projection = nn.Parameter(torch.empty(GPT2Config().n_embd, hidden_dim))
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
  
  def forward(self, x1, x2):
    local = self.embed(x1)
    external = self.embed(x2)

    local = self.fc1(local.view(local.shape[0], -1))
    external = self.fc1(external.view(external.shape[0], -1))


    norm_local = local / local.norm(dim=1, keepdim=True)
    norm_external = external / external.norm(dim=1, keepdim=True)
    logit_scale = self.logit_scale.exp()
    local_logits = logit_scale * norm_local @ norm_external.t()
    external_logits = local_logits.t()

    return local_logits, external_logits
  def __str__(self):
    return "SimpleModel"

class Attention(nn.Module):
    def __init__(self,hidden_dim, output_dim=1024, attn_dim=512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.attn_dim = attn_dim

        self.w1 = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.tanh = nn.Tanh()
        self.w2 = nn.Linear(self.output_dim, self.attn_dim)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(self.attn_dim,1)
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, encoder_outputs):

        src_len = encoder_outputs.shape[1]

        output_fw = encoder_outputs[:,:,0:self.hidden_dim]
        output_bw = encoder_outputs[:,:,self.hidden_dim:]

        hidden_states = torch.cat((output_fw, output_bw),-1)

        # Obtaining the attention weights
        weighted_states = self.w1(hidden_states)
        activated_states = self.tanh(weighted_states)
        score_weights = self.w2(activated_states)
        attention_weights = self.softmax(score_weights)
        
        # Applying attention to the matrix with hidden states
        attentional_vector = torch.bmm(torch.transpose(attention_weights,2,1),hidden_states)   
        attentional_vector = self.fc(torch.transpose(attentional_vector,2,1)).squeeze(2)
        attentional_vector = self.leakyRelu(attentional_vector)

        return attentional_vector

class AttentionBiGRU(torch.nn.Module):
  def __init__(self, tokenizer, embed_dim, hidden_dim=256):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.embed = GPT2_WordEncoder(tokenizer)
    self.gru = nn.GRU(embed_dim, hidden_dim//2, bidirectional=True)
    self.attention = Attention(hidden_dim//2)
    self.distance = nn.CosineSimilarity(dim=1)
    self.sigmoid = nn.Sigmoid()
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.GRU):
      for param_name,weights in module.named_parameters():
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

    norm_local = attn_l / attn_l.norm(dim=1, keepdim=True)
    norm_external = attn_e / attn_e.norm(dim=1, keepdim=True)

    logit_scale = self.logit_scale.exp()
    local_logits = logit_scale * norm_local @ norm_external.t()
    external_logits = local_logits.t()


    return local_logits, external_logits
  def __str__(self):
    return "AttentionBiGRU"

def contrastive_loss(loss_l, loss_e, labels):
  total_loss = (
            F.cross_entropy(loss_l, labels) +
            F.cross_entropy(loss_e, labels)
        ) / 2
  return total_loss

def validation_metrics (model, loader):
  model.eval()
  correct = 0
  total = 0
  sum_loss = 0.0
  for j, (local, external, y) in enumerate(loader):
    loss_l, loss_e = model(local,external)
    labels = torch.arange(loss_l.shape[0]).cuda()
    loss = contrastive_loss(loss_l, loss_e, labels)
    
    probs = loss_l.softmax(dim=-1)
    y_pred = torch.argmax(probs,axis=1)

    for l in labels:
        if probs[l][y_pred[l]] < 0.5:
          y_pred[l] = -1

    correct += (y_pred == labels).float().sum()
    sum_loss += loss.item()*y.shape[0]
    total += y.shape[0]

    return sum_loss/total, correct/total#, batch_predictions, batch_labels#, correct_0, correct_1, total_0, total_1, total

def train_model(model, epochs=1000, lr=0.00001):
  model.train()
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.00001)
  best_test_loss = float('inf')
  for i in range(epochs):
    model.train()
    sum_loss = 0.0
    total = 0
    correct = 0
    for j, (local, external, y) in enumerate(train_loader):
      loss_l, loss_e = model(local,external)
      labels = torch.arange(loss_l.shape[0]).cuda()
      loss = contrastive_loss(loss_l, loss_e, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      probs = loss_l.softmax(dim=-1)
      y_pred = torch.argmax(probs, axis=1)
      for l in labels:
        if probs[l][y_pred[l]] < 0.5:
          y_pred[l] = -1


      correct += (y_pred == labels).float().sum()
      sum_loss += loss.item()*y.shape[0]
      total += y.shape[0]
      
    test_loss, test_acc = validation_metrics(model, test_loader)
    print("epoch %d train loss %.3f, train acc %.3f, test loss %.3f, test acc %.3f" % 
            (i, sum_loss/total, correct/total, test_loss, test_acc))
    
    if test_loss < best_test_loss:
      best_test_loss = test_loss
      torch.save(model.state_dict(), str(model)+'-bestcheckpoint.pt')
print("building model")


########## Build & Train Models ###############

model = SimpleModel(googleTrain.tokenizer,GPT2Config().n_embd)
model.cuda()
print("training")
train_model(model)

model = AttentionBiGRU(googleTrain.tokenizer,GPT2Config().n_embd)
model.cuda()
train_model(model)


################# Eval ################

def createHardDatasets(testData, batchSize, folder="google"):
  df_p_local = list(testData['local_tuple'])
  df_p_external = list(testData['external_tuple'])
  df_p_rel = list(testData['relevant'])

  df_n = pd.read_csv(folder+'/negative.csv')

  df_merged = {}
  df_merged['local_tuple'] = list()
  df_merged['external_tuple'] = list()
  df_merged['relevant'] = list()

  for i in range(len(testData['local_tuple'])):
    pos_local = df_p_local[i]
    pos_external = df_p_external[i]
    pos_rel = df_p_rel[i]

    negative_samples = df_n.loc[df_n['local_tuple'] == pos_local][0:(batchSize-1)] 
    if(len(negative_samples) == (batchSize-1)):

      df_merged['local_tuple'].append(pos_local)
      df_merged['external_tuple'].append(pos_external)
      df_merged['relevant'].append(pos_rel)

      df_merged['local_tuple'].extend(list(negative_samples['local_tuple']))
      df_merged['external_tuple'].extend(list(negative_samples['external_tuple']))
      df_merged['relevant'].extend(list(negative_samples['relevant']))

  return df_merged

hardTestData = createHardDatasets(googleTestData, B, "google")
hardTest = TupleData(hardTestData, 'gpt2')
hard_test_loader = DataLoader(hardTest, batch_size=B, shuffle=False, collate_fn=lambda b: pad_collate(b, hardTest.tokenizer, min(512, hardTest.max_seq_len)))

print("loading bigru")
model = AttentionBiGRU(googleTrain.tokenizer,GPT2Config().n_embd)
model.cuda()
model.load_state_dict(torch.load(str(model)+'-bestcheckpoint.pt'))
model.eval()

print("loading simple")
baseline = SimpleModel(googleTrain.tokenizer,GPT2Config().n_embd)
baseline.cuda()
baseline.load_state_dict(torch.load(str(baseline)+'-bestcheckpoint.pt'))
baseline.eval()

##### Train Eval ####

print("Train BiGru")

train_vals = {}
train_vals['correct'] = list()
train_vals['incorrect'] = list()

correct = 0
total = 0
loss = 0
y_true = []
y_predict = []
for j, (local, external, y) in enumerate(train_loader):
  loss_l, loss_e = model(local,external)
  labels = torch.arange(loss_l.shape[0]).cuda()
  loss += contrastive_loss(loss_l, loss_e, labels).item()*y.shape[0]
  probs = loss_l.softmax(dim=-1)
  y_pred = torch.argmax(probs, axis=1)

  y_predict.extend(y_pred.cpu().detach().numpy())
  y_true.extend(labels.cpu().detach().numpy())

  for l in range(len(labels)):
    if y_pred[l] == labels[l]:
      train_vals['correct'].append(probs[l][y_pred[l]].cpu().detach().numpy())
    else:
      train_vals['incorrect'].append(probs[l][y_pred[l]].cpu().detach().numpy())

  correct += (y_pred == labels).float().sum()
  total += y.shape[0]
#print(classification_report(y_true, y_predict))
print('Train Accuracy: ', (correct/total))
print('Train Loss: ', (loss/total))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(train_vals['correct'], bins=10)
axs[1].hist(train_vals['incorrect'], bins=10)
axs[0].set_title("Correct Prediction")
axs[1].set_title("Incorrect Prediction")
axs[0].set_xlabel("Predicted Probability")
axs[1].set_xlabel("Predicted Probability")
axs[0].set_ylabel("Count")
fig.savefig("train-bigru.png")

print("Train Baseline")

train_vals = {}
train_vals['correct'] = list()
train_vals['incorrect'] = list()

correct = 0
total = 0
loss = 0
y_true = []
y_predict = []
for j, (local, external, y) in enumerate(train_loader):
  loss_l, loss_e = baseline(local,external)
  labels = torch.arange(loss_l.shape[0]).cuda()
  loss += contrastive_loss(loss_l, loss_e, labels).item()*y.shape[0]
  probs = loss_l.softmax(dim=-1)
  y_pred = torch.argmax(probs, axis=1)

  y_predict.extend(y_pred.cpu().detach().numpy())
  y_true.extend(labels.cpu().detach().numpy())

  for l in range(len(labels)):
    if y_pred[l] == labels[l]:
      train_vals['correct'].append(probs[l][y_pred[l]].cpu().detach().numpy())
    else:
      train_vals['incorrect'].append(probs[l][y_pred[l]].cpu().detach().numpy())

  correct += (y_pred == labels).float().sum()
  total += y.shape[0]
#print(classification_report(y_true, y_predict))
print('Train Accuracy: ', (correct/total))
print('Train Loss: ', (loss/total))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(train_vals['correct'], bins=10)
axs[1].hist(train_vals['incorrect'], bins=10)
axs[0].set_title("Correct Prediction")
axs[1].set_title("Incorrect Prediction")
axs[0].set_xlabel("Predicted Probability")
axs[1].set_xlabel("Predicted Probability")
axs[0].set_ylabel("Count")
fig.savefig("train-simple.png")



##### Test Eval ####
print("Test BiGru")

test_vals = {}
test_vals['correct'] = list()
test_vals['incorrect'] = list()

correct = 0
total = 0
loss = 0
y_true = []
y_predict = []
for j, (local, external, y) in enumerate(test_loader):
  loss_l, loss_e = model(local,external)
  labels = torch.arange(loss_l.shape[0]).cuda()
  loss += contrastive_loss(loss_l, loss_e, labels).item()*y.shape[0]
  probs = loss_l.softmax(dim=-1)
  y_pred = torch.argmax(probs, axis=1)

  y_predict.extend(y_pred.cpu().detach().numpy())
  y_true.extend(labels.cpu().detach().numpy())

  for l in range(len(labels)):
    if y_pred[l] == labels[l]:
      test_vals['correct'].append(probs[l][y_pred[l]].cpu().detach().numpy())
    else:
      test_vals['incorrect'].append(probs[l][y_pred[l]].cpu().detach().numpy())

  correct += (y_pred == labels).float().sum()
  total += y.shape[0]
#print(classification_report(y_true, y_predict))
print('Test Accuracy: ', (correct/total))
print('Test Loss: ', (loss/total))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(test_vals['correct'], bins=10)
axs[1].hist(test_vals['incorrect'], bins=10)
axs[0].set_title("Correct Prediction")
axs[1].set_title("Incorrect Prediction")
axs[0].set_xlabel("Predicted Probability")
axs[1].set_xlabel("Predicted Probability")
axs[0].set_ylabel("Count")
fig.savefig("test-bigru.png")

print("Test Baseline")
test_vals = {}
test_vals['correct'] = list()
test_vals['incorrect'] = list()

correct = 0
total = 0
loss = 0
y_true = []
y_predict = []
for j, (local, external, y) in enumerate(test_loader):
  loss_l, loss_e = baseline(local,external)
  labels = torch.arange(loss_l.shape[0]).cuda()
  loss += contrastive_loss(loss_l, loss_e, labels).item()*y.shape[0]
  probs = loss_l.softmax(dim=-1)
  y_pred = torch.argmax(probs, axis=1)

  y_predict.extend(y_pred.cpu().detach().numpy())
  y_true.extend(labels.cpu().detach().numpy())

  for l in range(len(labels)):
    if y_pred[l] == labels[l]:
      test_vals['correct'].append(probs[l][y_pred[l]].cpu().detach().numpy())
    else:
      test_vals['incorrect'].append(probs[l][y_pred[l]].cpu().detach().numpy())

  correct += (y_pred == labels).float().sum()
  total += y.shape[0]
#print(classification_report(y_true, y_predict))
print('Test Accuracy: ', (correct/total))
print('Test Loss: ', (loss/total))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(test_vals['correct'], bins=10)
axs[1].hist(test_vals['incorrect'], bins=10)
axs[0].set_title("Correct Prediction")
axs[1].set_title("Incorrect Prediction")
axs[0].set_xlabel("Predicted Probability")
axs[1].set_xlabel("Predicted Probability")
axs[0].set_ylabel("Count")
fig.savefig("test-simple.png")

##### Hard Negative Eval #####

print("Bigru Hard Negative")

hard_test_vals = {}
hard_test_vals['correct'] = list()
hard_test_vals['incorrect'] = list()
hard_test_vals['batch_probs'] = list()

correct = 0
total = 0
loss = 0
y_true = []
y_predict = []
for j, (local, external, y) in enumerate(hard_test_loader):
  assert(z == 0 for z in y[1:B])
  assert(y[0].item() == 1) 
  loss_l, loss_e = model(local,external)
  labels = torch.arange(loss_l.shape[0]).cuda()
  loss += contrastive_loss(loss_l, loss_e, labels).item()
  probs = loss_l.softmax(dim=-1)
  y_pred = torch.argmax(probs, axis=1)

  hard_test_vals['batch_probs'].append(probs[0].cpu().detach().numpy())

  y_predict.append(y_pred[0].cpu().detach().numpy())
  y_true.append(labels[0].cpu().detach().numpy())
  

  if y_pred[0] == labels[0]:
      hard_test_vals['correct'].append(probs[0][y_pred[0]].cpu().detach().numpy())
  else:
    hard_test_vals['incorrect'].append(probs[0][y_pred[0]].cpu().detach().numpy())


  correct += (y_pred[0] == labels[0]).float().sum()
  total += 1
#print(classification_report(y_true, y_predict))
print('Hard Test Accuracy: ', (correct/total))
print('Hard Loss Accuracy: ', (loss/total))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(hard_test_vals['correct'], bins=10)
axs[1].hist(hard_test_vals['incorrect'], bins=10)
axs[0].set_title("Correct Prediction")
axs[1].set_title("Incorrect Prediction")
axs[0].set_xlabel("Predicted Probability")
axs[1].set_xlabel("Predicted Probability")
axs[0].set_ylabel("Count")
fig.savefig("hard-neg-bigru.png")


print("Hard Negative Baseline")

hard_test_vals = {}
hard_test_vals['correct'] = list()
hard_test_vals['incorrect'] = list()
hard_test_vals['batch_probs'] = list()

correct = 0
total = 0
loss = 0
y_true = []
y_predict = []
for j, (local, external, y) in enumerate(hard_test_loader):
  assert(z == 0 for z in y[1:B])
  assert(y[0].item() == 1) 
  loss_l, loss_e = baseline(local,external)
  labels = torch.arange(loss_l.shape[0]).cuda()
  loss += contrastive_loss(loss_l, loss_e, labels).item()
  probs = loss_l.softmax(dim=-1)
  y_pred = torch.argmax(probs, axis=1)

  hard_test_vals['batch_probs'].append(probs[0].cpu().detach().numpy())

  y_predict.append(y_pred[0].cpu().detach().numpy())
  y_true.append(labels[0].cpu().detach().numpy())
  

  if y_pred[0] == labels[0]:
      hard_test_vals['correct'].append(probs[0][y_pred[0]].cpu().detach().numpy())
  else:
    hard_test_vals['incorrect'].append(probs[0][y_pred[0]].cpu().detach().numpy())


  correct += (y_pred[0] == labels[0]).float().sum()
  total += 1
#print(classification_report(y_true, y_predict))
print('Hard Test Accuracy: ', (correct/total))
print('Hard Loss Accuracy: ', (loss/total))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(hard_test_vals['correct'], bins=10)
axs[1].hist(hard_test_vals['incorrect'], bins=10)
axs[0].set_title("Correct Prediction")
axs[1].set_title("Incorrect Prediction")
axs[0].set_xlabel("Predicted Probability")
axs[1].set_xlabel("Predicted Probability")
axs[0].set_ylabel("Count")
fig.savefig("hard-neg-simple.png")


