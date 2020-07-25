from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, text, category, tokenizer, max_len):
        self.text = text
        self.category = category
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        category = self.category[item]

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

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category': torch.tensor(category, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = CustomDataset(
    text=df['text'].to_numpy(),
    category=df['category'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )