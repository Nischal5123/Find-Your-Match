from sentence_transformers import SentenceTransformer, models,InputExample,losses
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import evaluation

#Ref:https://www.sbert.net/docs/training/overview.html

###########################
# BASIC IDEA
###########################

# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
#
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])#output will be 256 dimension
# train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# train_loss = losses.CosineSimilarityLoss(model)
# train_loss = losses.CosineSimilarityLoss(model)
#
# sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
# sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
# scores = [0.3, 0.6, 0.2]
#
# evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=1)
#
# print(evaluator.scores)


#################################################################################################################################################################

"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.
Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/'  # REPLACE THIS WITH OUR DATASET "TUPLE-1", "TUPLE-2" , "SCORE"

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)




# Read the dataset
model_name = 'nli-distilroberta-base-v2'
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read Drug dataset")

train_samples = []
dev_samples = []
test_samples = []

positive_csv_reader =pd.read_csv(sts_dataset_path + "positive.csv")
negative_csv_reader =pd.read_csv(sts_dataset_path + "negative.csv")

for idx,row in positive_csv_reader.iterrows():
    score = float(row['relevant'])  # Normalize score to range 0 ... 1
    inp_example = InputExample(texts=[row['local_tuple'], row['external_tuple']], label=score)

    if row['split'] == 'dev':
        dev_samples.append(inp_example)
    elif row['split'] == 'test':
        test_samples.append(inp_example)
    else:
        train_samples.append(inp_example)


for idx,row in negative_csv_reader.iterrows():
    score = float(row['relevant'])  # Normalize score to range 0 ... 1
    inp_example = InputExample(texts=[row['local_tuple'], row['external_tuple']], label=score)

    if row['split'] == 'dev':
        dev_samples.append(inp_example)
    elif row['split'] == 'test':
        test_samples.append(inp_example)
    else:
        train_samples.append(inp_example)



train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read Drug dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
