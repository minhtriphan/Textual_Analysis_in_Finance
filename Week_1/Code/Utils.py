import re
from typing import Optional, List

import contractions    # Don't forget to install it via `pip install contractions`

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def readfile(f: str) -> str:
    '''
    This function reads the content of a .txt file.
    Args:
      - f (str): the file name with full directory.
    Output:
      - File content (str)
    '''
    with open(f, 'r') as f:
        content = f.read()
    return content

def lowercase(txt: str) -> str:
    '''
    This function converts the text into lowercase.
    Args:
      - txt (str): The raw file content.
    Output:
      - The text after lowercasing (str).
    '''
    return txt.lower()

def fix_contraction(txt: str) -> str:
    '''
    This function fixes contractions, which are terms like I'll, We're, he's, etc., to the full forms, such as I will, We are, etc.
    WARNINGS: This function should be implemented before removing special characters (punctuations)!!!
    Args:
      - The text with corrected contractions.
    '''
    return ' '.join([contractions.fix(i) for i in txt.split()])

def special_character_removal(txt: str, keep: Optional[str] = None, remove_number: bool = False) -> str:
    '''
    This function removes punctuations of a text.
    Args:
      - txt (str): The text (could be after lowercasing).
      - keep (str): A string storing punctuations to keep. For example: ',.()'. Default: None, which means exclude all punctuations.
      - remove_number (bool): Whether remove numbers or not.
    Output:
      - The text after removing punctations.
    '''
    if remove_number:
            # This code substitute all characters that are not (^) alphabetic (A-Z), numbers (0-9), and those defined in `keep` (if any)
        txt = re.sub(f'[^A-Za-z{keep}]+', ' ', txt)
    else:
        txt = re.sub(f'[^A-Za-z0-9{keep}]+', ' ', txt)
    return txt

def stopword_removal(txt: str, stopword_list: List = stopwords.words('english'), remove_single_letters: bool = False) -> str:
    '''
    This function removes stopwords, which are defined based on the `nltk` Python package.
    Args:
      - txt (str): The text after lowercasing and removing special characters.
      - stopword_list (list): A list of pre-defined stopwords.
      - remove_single_letters (bool): Whether to remove single letters, such as a, b, c, etc., or not. Default is False.
    Output:
      - The text after removing stopwords.
    '''
    if remove_single_letters:
        _stopword_list = stopword_list + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.lower())
    else:
        _stopword_list = stopword_list
    return ' '.join([word for word in txt.split() if word not in _stopword_list])
