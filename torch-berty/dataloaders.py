from torch.utils.data import Dataset, DataLoader
import torch
class CustomDataset(Dataset):
    '''
    Class for creating a generic Dataset object for pytorch classification
    '''
    def __init__(self, text, category, tokenizer, max_len):
        '''

        :param text: Text field  of the dataset
        :param category: Category field of the dataset
        :param tokenizer: Tokenizer selected for the model
        :param max_len: MAX LEN for the dataset
        '''
        self.text = text
        self.category = category
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        category = self.category[item]

        #tokenizing the input
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        #return data in order BERT requires
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category': torch.tensor(category, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size, text_field, category_field):
    '''
    Returns data loader for a given dataframe with specs
    :param df: The dataframe for which data loader is made
    :param tokenizer: Tokenizer to be used
    :param max_len: MAX LEN selected for the tokenizer
    :param batch_size: batch size selected
    :param text_field: the text field
    :param category_field: label field in the dataset
    :return:
    '''
    ds = CustomDataset(
    text=df[text_field].to_numpy(),
    category=df[category_field].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
    )

    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
    )