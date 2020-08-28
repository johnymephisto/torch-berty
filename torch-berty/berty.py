# -*- coding: utf-8 -*-
"""Torch Berty Project

This module aims at bringing the power of bert to the masses with easy to use
entry point to train and get a production ready model.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        from berty import Berty
        bert_model = Berty()
        bert_model.load_dataset("train.csv", 'text', 'target')
        bert_model.fit()


"""

__version__ = '0.1'

import logging
logging.basicConfig(level=logging.ERROR)

import pandas as pd

from dataloaders import create_data_loader
from model import Model

import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split


class Berty:
    '''
    Main class that acts as the interface
    '''

    def __init__(self, pre_trained_model='bert-base-uncased', batch_size=16):
        '''
        Constructor
        :param pre_trained_model: specify the BERT pretrained model to be used, uses 'bert-base-uncased'
                                    by default
        :param batch_size: uses 16 by default
        '''
        self.PRE_TRAINED_MODEL_NAME = pre_trained_model
        self.RANDOM_SEED  =  42
        self.BATCH_SIZE = batch_size

    def load_dataset(self, csv_file, text_field , label_field):
        '''
        Load the dataset you want to do training on

        :param csv_file: The dataset in csv format
        :param text_field: specify field name in which th text is present on which model has to be trained
        :param label_field: specify the target label name
        :return:
        '''

        self.text_field = text_field
        self.label_field = label_field

        #load the  dataset with only the required fields
        self.dataset = pd.read_csv(csv_file, usecols=[text_field, label_field])

        print("Dropping NaN values from the dataset")
        self.dataset = self.dataset.dropna()

        print(self.dataset.info())

    def fit(self):
        '''
        Use this function to train the model
        :return: trained model
        '''

        #Store the category values to encoded numeric format and vice versa for prediction time
        cat_to_num = {item: idx for idx, item in enumerate(self.dataset[self.label_field].unique())}
        num_to_cat = {idx: item for idx, item in enumerate(self.dataset[self.label_field].unique())}

        #Label Encoding
        to_nums = lambda cat : cat_to_num[cat]

        self.dataset[self.label_field] = self.dataset[self.label_field].apply(to_nums)

        #Initialize the tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        #Calculate max token length that can occur
        token_lens = []
        for txt in self.dataset[self.text_field]:
            tokens = tokenizer.encode(txt, max_length=512)
            token_lens.append(len(tokens))

        MAX_LEN = max(token_lens)  if max(token_lens)<512  else 512
        print('Max Len set as  {}'.format(MAX_LEN))

        #Split the train and validation set
        df_train, df_test = train_test_split(self.dataset, test_size=0.1, random_state=self.RANDOM_SEED)

        self.dataset[self.label_field].to_numpy()

        #Initialise the data loaders for train and validation
        train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, self.BATCH_SIZE, self.text_field, self.label_field)
        test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, self.BATCH_SIZE, self.text_field, self.label_field)

        #Create PL model object
        bert_finetuner = Model(len(cat_to_num), self.PRE_TRAINED_MODEL_NAME, train_data_loader=train_data_loader,
                               test_data_loader=test_data_loader)

        #Initialise PL trainer object
        trainer = pl.Trainer(gpus=1)

        #Train and validate the model
        trainer.fit(bert_finetuner)

        #TODO saving the model and generating script for inference

        return bert_finetuner


if  __name__ == "__main__":

    bert_model = Berty()
    bert_model.load_dataset("/home/mephist0/projects//twitter_data/train.csv", 'text', 'target')
    bert_model.fit()