# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.ERROR)

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""## Data Exploration

We'll load the Google Play app reviews dataset, that we've put together in the previous part:
"""

df = pd.read_csv("/home/mephist0/projects//twitter_data/train.csv")
print(df.head())


df['pagetitle'] = df['text']
df['Category'] = df['target']

df = df[['pagetitle', 'Category']]



print(df['Category'].unique())

"""We have about 16k examples. Let's check for missing values:"""



# df.info()
df = df.dropna()
print(df.info())


cat_to_num = {item:idx for idx, item in enumerate(df.Category.unique())}
num_to_cat = {idx:item for idx, item in enumerate(df.Category.unique())}

print(cat_to_num)

def to_nums(cat):
  return cat_to_num[cat]

df['Category'] = df.Category.apply(to_nums)


PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

# from transformers import AutoModel, AutoTokenizer, __version__

# print(__version__)
# model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
# tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

"""We'll use this text to understand the tokenization process:"""

sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'

"""Some basic operations can convert the text to tokens and tokens to unique integers (ids):"""

tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')


encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',
  truncation=True
)



token_lens = []

for txt in df.pagetitle:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))


"""Most of the reviews seem to contain less than 128 tokens, but we'll be on the safe side and choose a maximum length of 160."""

MAX_LEN = 160

"""We have all building blocks required to create a PyTorch dataset. Let's do it:"""

class PageTitleDataset(Dataset):

  def __init__(self, title, category, tokenizer, max_len):
    self.title = title
    self.category = category
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.title)
  
  def __getitem__(self, item):
    title = str(self.title[item])
    category = self.category[item]

    encoding = self.tokenizer.encode_plus(
      title,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True
    )

    return {
      'title_text': title,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'category': torch.tensor(category, dtype=torch.long)
    }

"""The tokenizer is doing most of the heavy lifting for us. We also return the review texts, so it'll be easier to evaluate the predictions from our model. Let's split the data:"""

df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

df_train.shape, df_val.shape, df_test.shape

df.Category.to_numpy()

"""We also need to create a couple of data loaders. Here's a helper function to do it:"""

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = PageTitleDataset(
    title=df.pagetitle.to_numpy(),
    category=df.Category.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

"""Let's have a look at an example batch from our training data loader:"""

data = next(iter(train_data_loader))
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['category'].shape)

"""## Sentiment Classification with BERT and Hugging Face

There are a lot of helpers that make using BERT easy with the Transformers library. Depending on the task you might want to use [BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification), [BertForQuestionAnswering](https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering) or something else. 

But who cares, right? We're *hardcore*! We'll use the basic [BertModel](https://huggingface.co/transformers/model_doc/bert.html#bertmodel) and build our sentiment classifier on top of it. Let's load the model:
"""

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

"""And try to use it on the encoding of our sample text:"""

last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'], 
  attention_mask=encoding['attention_mask']
)

"""The `last_hidden_state` is a sequence of hidden states of the last layer of the model. Obtaining the `pooled_output` is done by applying the [BertPooler](https://github.com/huggingface/transformers/blob/edf0582c0be87b60f94f41c659ea779876efc7be/src/transformers/modeling_bert.py#L426) on `last_hidden_state`:"""

last_hidden_state.shape

"""We have the hidden state for each of our 32 tokens (the length of our example sequence). But why 768? This is the number of hidden units in the feedforward-networks. We can verify that by checking the config:"""

bert_model.config.hidden_size

"""You can think of the `pooled_output` as a summary of the content, according to BERT. Albeit, you might try and do better. Let's look at the shape of the output:"""

pooled_output.shape
from sklearn.metrics import accuracy_score

loss_fn = nn.CrossEntropyLoss().to(device)

"""We can use all of this knowledge to create a classifier that uses the BERT model:"""

class SentimentClassifier(pl.LightningModule):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

  def training_step(self, batch, batch_nb):


    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["category"].to(device)

    outputs = self(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    tensorboard_logs = {'train_loss': loss}
    return {'loss': loss, 'log': tensorboard_logs}

  def validation_step(self, batch, batch_nb):
    # batch
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    targets = batch["category"].to(device)

    outputs = self(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)

    loss = loss_fn(outputs, targets)

    val_acc = (torch.sum(preds == targets).double())/len(batch)
    val_acc = torch.tensor(val_acc)

    return {'val_loss': loss, 'val_acc': val_acc}

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

    tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
    return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
    return optimizer

  def train_dataloader(self):
    return train_data_loader

  def val_dataloader(self):
    return val_data_loader


bert_finetuner = SentimentClassifier(2)

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)
trainer.fit(bert_finetuner)
"""Our classifier delegates most of the heavy lifting to the BertModel. We use a dropout layer for some regularization and a fully-connected layer for our output. Note that we're returning the raw output of the last layer since that is required for the cross-entropy loss function in PyTorch to work.

This should work like any other PyTorch model. Let's create an instance and move it to the GPU:
"""

model = SentimentClassifier(len(cat_to_num.keys()))
model = model.to(device)

"""We'll move the example batch of our training data to the GPU:"""

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

"""To get the predicted probabilities from our trained model, we'll apply the softmax function to the outputs:"""

F.softmax(model(input_ids, attention_mask), dim=1)

"""### Training

To reproduce the training procedure from the BERT paper, we'll use the [AdamW](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#adamw) optimizer provided by Hugging Face. It corrects weight decay, so it's similar to the original paper. We'll also use a linear scheduler with no warmup steps:
"""

EPOCHS = 5

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

"""How do we come up with all hyperparameters? The BERT authors have some recommendations for fine-tuning:

- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4

We're going to ignore the number of epochs recommendation but stick with the rest. Note that increasing the batch size reduces the training time significantly, but gives you lower accuracy.

Let's continue with writing a helper function for training our model for one epoch:
"""

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["category"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

"""Training the model should look familiar, except for two things. The scheduler gets called every time a batch is fed to the model. We're avoiding exploding gradients by clipping the gradients of the model using [clip_grad_norm_](https://pytorch.org/docs/stable/nn.html#clip-grad-norm).

Let's write another one that helps us evaluate the model on a given data loader:
"""

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["category"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

"""Using those two, we can write our training loop. We'll also store the training history:"""


history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc



"""Note that we're storing the state of the best model, indicated by the highest validation accuracy.

Whoo, this took some time! We can look at the training vs validation accuracy:
"""

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);

"""The training accuracy starts to approach 100% after 10 epochs or so. You might try to fine-tune the parameters a bit more, but this will be good enough for us.

Don't want to wait? Uncomment the next cell to download my pre-trained model:

## Evaluation

So how good is our model on predicting sentiment? Let's start by calculating the accuracy on the test data:
"""

test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

test_acc.item()

"""The accuracy is about 1% lower on the test set. Our model seems to generalize well.

We'll define a helper function to get the predictions from our model:
"""

def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["title_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["category"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

"""This is similar to the evaluation function, except that we're storing the text of the reviews and the predicted probabilities (by applying the softmax on the model outputs):"""

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)

"""Let's have a look at the classification report"""

print(classification_report(y_test, y_pred, target_names=cat_to_num.keys()))

"""We'll continue with the confusion matrix:"""

def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True ')
  plt.xlabel('Predicted ');

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=cat_to_num.keys(), columns=cat_to_num.keys())
show_confusion_matrix(df_cm)