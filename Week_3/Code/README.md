
# ELMo implementation

To use ELMo, first train the model by running the following code, make sure your working directory is `Textual_Analysis_in_Finance`

```
from glob import glob
from Week_3.Code.elmo_utils import build_vocab
from Week_3.Code.elmo_trainer import Trainer
from Week_3.Code.elmo_config import Config

# Load all files
all_files = glob('/content/Textual_Analysis_in_Finance/Data/FinancialStatements/*.txt')

# Build the vocabulary
vocab = build_vocab(all_files, min_freq = 5)

# Initialize the trainer
config = Config()
config.nepochs = 5    # Change here the number of epochs the model is trained
trainer = Trainer(config, vocab)

# Train
trainer.train(all_files)
```

After training, we can use the model to infer embedding of a new/unseen text

```
# Infer embedding for a new text
sample_text = 'income increase'
text_embedding = trainer.infer(sample_text)
text_embedding
```
