from typing import List, Dict
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from Week_3.Code.elmo_model import ELMo
from Week_3.Code.elmo_dataset import ELMoDataset, Collator

class Trainer():
    def __init__(
        self,
        config,
        vocab: Dict,
    ):
        self.config = config
        self.vocab = vocab

    def _prepare_dataset(self, filenames: List[str]):
        return ELMoDataset(filenames = filenames, vocab = self.vocab)

    def _prepare_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size = self.config.batch_size,
            num_workers = self.config.num_workers,
            collate_fn = Collator(self.config.max_length, padding = 'max_length', truncation = True)
        )

    def _prepare_model(self):
        return ELMo(len(self.vocab))

    def _prepare_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr = self.config.lr)

    def _encode_text(self, text):
        tokens = text.lower().split()[:self.config.max_length]
        ids = []
        for word in tokens:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.append(self.vocab['<unk>'])
        return torch.tensor(ids).unsqueeze(0)

    def train(self, filenames):
        # Prepare the dataset
        dataset = self._prepare_dataset(filenames)
        # Prepare the dataloader
        dataloader = self._prepare_dataloader(dataset)
        # Prepare the model
        model = self._prepare_model()
        # Prepare the optimizer
        optimizer = self._prepare_optimizer(model)

        for epoch in tqdm(range(self.config.nepochs)):
            total_loss = 0
            for batch in dataloader:
                logits, pooled_output, loss = model(batch['input_ids'], attention_mask = batch['attention_mask'])

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch}: {total_loss:.2f}')

        self.model = model    # After training, save the model as an attribute of the Trainer. NOTICE: This line of code will NOT save the model to hard drive!!!

    def infer(self, text):
        # Prepare the model
        # If the model is trained already, load its weights. For simplicity, assume that we use an un-trained model to infer embedding of this text
        try:
            model = self.model
        except:
            model = self._prepare_model()
        # Encode the text
        input_ids = self._encode_text(text)
        with torch.no_grad():
            _, pooled_output, _ = model(input_ids)
        return pooled_output.numpy().flatten()
