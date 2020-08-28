import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn


class Model(pl.LightningModule):
    '''
    Model Architecture definition
    Made using Pytorch Lightning framework

    contains methods for train and validation
    '''

    def __init__(self, n_classes, pre_trained_model_name, train_data_loader, test_data_loader):
        '''

        :param n_classes: Number of classes in the data set
        :param pre_trained_model_name: any of the pre-trained BERT model types
        :param train_data_loader: data loader for training data
        :param test_data_loader: data loader for testing data
        '''
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def setup(self, stage: str):
        # checking if gpu is available and setting default device to place tensors
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # CE loss is selected as the loss function
        self.loss_fn = nn.CrossEntropyLoss().to(self.dev)

    def forward(self, input_ids, attention_mask):
        '''
        Forward propagation of the model

        :param input_ids: ids from the tokenizer for a data point
        :param attention_mask: attention mask from the tokenizer
        :return: output of forward prop
        '''

        #output from the pretrained bert model
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        #output from dropout layer
        output = self.drop(pooled_output)

        #output from the final output layer
        return self.out(output)

    def training_step(self, batch, batch_nb):
        '''
        Training function

        :param batch: batch of data
        :param batch_nb: batch number
        :return: loss
        '''

        #All input tensors placed in device
        input_ids = batch["input_ids"].to(self.dev)
        attention_mask = batch["attention_mask"].to(self.dev)
        targets = batch["category"].to(self.dev)

        #one forward prop step
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        #predictions for current step
        _, preds = torch.max(outputs, dim=1)

        #loss calculated for current step
        loss = self.loss_fn(outputs, targets)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        '''
        Validation function
        :param batch: batch of validation data
        :param batch_nb: batch number
        :return: loss and accuracy metrics
        '''

        # All input tensors placed in device
        input_ids = batch["input_ids"].to(self.dev)
        attention_mask = batch["attention_mask"].to(self.dev)
        targets = batch["category"].to(self.dev)

        ##one forward prop step
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # predictions for current step
        _, preds = torch.max(outputs, dim=1)

        # loss calculated for current step
        loss = self.loss_fn(outputs, targets)

        #TODO fix bug in acc calculation
        val_acc = (torch.sum(preds == targets).double()) / len(batch)
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        '''
        Things to do after validation step ends

        :param outputs: collection of loss and acc metric across batches
        :return: consolidated loss and acc metrics
        '''
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        '''
        Optimiser configuration

        :return: optimizer
        '''

        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        return optimizer

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.test_data_loader