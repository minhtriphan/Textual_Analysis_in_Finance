import random
from typing import List, Dict
import torch
from torch.utils.data import Dataset

from Week_3.Code.elmo_utils import text_normalization

class ELMoDataset(Dataset):
    def __init__(
        self, 
        filenames: List[str], 
        vocab: Dict
    ):
        self.filenames = filenames
        self.vocab = vocab

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # For each file, read and clean
        content = text_normalization(self.filenames[idx])

        # Split the content into many sentences
        sentences = content.split('.')

        # Sample only ONE sentence at a time
        sentence = random.choice(sentences)
        
        # Tokenize the sentence
        sentence = sentence.split()

        # Encode words in the sentence
        token_ids = []

        for word in sentence:
            if word in self.vocab:
                token_id = self.vocab[word]
            else:
                token_id = self.vocab['<unk>']
            token_ids.append(token_id)
        
        return token_ids

class Collator():
    def __init__(
        self,
        max_len: int,
        padding: str = 'max_length',
        truncation: bool = False,
    ):
        self.max_len = max_len
        self.padding = padding
        self.truncation = truncation

    def _pad_or_truncate(self, token_ids: List[int]):
        if len(token_ids) < self.max_len:
            # Now we generate the mask and pad
            attention_mask = [1] * len(token_ids) + [0] * (self.max_len - len(token_ids))
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))    # 0 is the padding token <pad>
            
        else:
            # Now, we generate the mask and truncate
            attention_mask = [1] * self.max_len
            token_ids = token_ids[:self.max_len]
        return token_ids, attention_mask

    def __call__(self, batch):
        token_ids = []
        attention_mask = []

        for item_token_ids in batch:
            item_token_ids, item_attention_mask = self._pad_or_truncate(item_token_ids)
            token_ids.append(item_token_ids)
            attention_mask.append(item_attention_mask)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype = torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype = torch.long)
        }
