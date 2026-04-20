# Re-use the code from Week 1 to normalize the text
from Week_1.Code.Utils import readfile, lowercase, special_character_removal, fix_contraction, stopword_removal, lemmatization
from collections import Counter
from typing import List, Dict
from tqdm.auto import tqdm

def text_normalization(f: str) -> str:
    '''
    This function reads a file and clean its content
    Args:
      - f (str): the file path
    Return:
      - A clean string
    '''
    text = readfile(f)
    text = lowercase(text)
    text = special_character_removal(text, keep = '.', remove_number = True)    # Keep the sentence structure
    text = fix_contraction(text)
    text = stopword_removal(text, remove_single_letters = True)
    text = lemmatization(text[:1_000_000])
    return text

def build_vocab(filenames: List[str], min_freq: int = 1, max_vocab_size: int = 1_000_000) -> Dict:
    '''
    This function reads all files in our data, clean it using the text_normalization function, then build the vocabulary
    Args:
      - filenames (list): A list of all file paths
      - min_freq (int): The lower threshold of which we discard words that appears less than this threshold in the corpus
      - max_vocab_size (int): The maximum vocabulary size
    Return:
      - The vocabulary in a Python dictionary
    '''
    # Initialize the counter, it will take care of counting unique words for us
    counter = Counter()

    for f in tqdm(filenames):
        # For each file in our data
        text = text_normalization(f)
        # Tokenize the clean text
        tokens = [i for i in text.split(' ') if i != '.']
        counter.update(tokens)
    
    # Initialize our vocabulary, it should have a padding token and an unknown token
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
    }

    for word, frequency in counter.most_common():
        if frequency < min_freq:
            # If the word appears less frequently than min_freq, ignore it
            continue
        else:
            if len(vocab) >= max_vocab_size:
                break
            else:
                vocab[word] = len(vocab)
    return vocab
