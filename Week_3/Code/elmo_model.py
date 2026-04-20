from typing import Union
import torch
from torch import nn
import  torch.nn.functional as F

class ELMo(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 300,
        hidden_size: int = 512
    ):
        super().__init__()

        # Embedding matrix
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # BiLSTM layer
        self.bilstm = nn.LSTM(
            embedding_size,
            hidden_size = hidden_size,
            bidirectional = True,
            batch_first = True,
            num_layers = 2,
            dropout = 0.1
        )

        # Output
        self.output = nn.Linear(hidden_size * 2, vocab_size)    # hidden_size * 2 is because of the Bi-LSTM

    def pooler(self, logits, attention_mask = None):
        if attention_mask is not None:
            return (logits * attention_mask.unsqueeze(-1)).sum(dim = 1) / attention_mask.sum(dim = 1, keepdim = True)
        else:
            return logits.mean(dim = 1)

    def loss_fn(self, logits, targets):
        # Forward - Predict future tokens using current and past tokens
        f_logits = logits[:, :-1, :]
        f_targets = targets[:, 1:]

        # Backward - Predict past tokens using current and future tokens
        b_logits = logits[:, 1:, :]
        b_targets = targets[:, :-1]

        loss_f = F.cross_entropy(
            f_logits.reshape(-1, f_logits.size(-1)),
            f_targets.reshape(-1),
            ignore_index = 0
        )

        loss_b = F.cross_entropy(
            b_logits.reshape(-1, b_logits.size(-1)),
            b_targets.reshape(-1),
            ignore_index = 0
        )

        return loss_f + loss_b

    def forward(self, input_ids: torch.Tensor, attention_mask: Union[torch.Tensor, None] = None):
        # Embed the input_ids
        embedding = self.embedding(input_ids)

        # Extract hidden state using the Bi-LSTM layer
        hidden_state, _ = self.bilstm(embedding)

        # Outputs
        logits = self.output(hidden_state)
        pooled_output = self.pooler(logits, attention_mask)
        loss = self.loss_fn(logits, input_ids)

        return logits, pooled_output, loss
