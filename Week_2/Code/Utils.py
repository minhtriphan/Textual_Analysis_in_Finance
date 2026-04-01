import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Union, List

class TextFeatureExtractor(object):
    def __init__(
        self,
        method: str = 'BoW'
    ):
        '''
        This object converts texts into vectors using different methods.
        Args:
          - method (str): The text extraction method, including 'BoW' (Bag-of-Words), 'TfIdf' (TF-IDF)
        '''
        self.method = method        

    def __call__(
        self,
        train_text: Union[pd.Series, List[str]],
        test_text: Union[pd.Series, List[str]]
    ):
        if self.method == 'bow':
            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(train_text)
            X_test = vectorizer.transform(test_text)
        elif self.method == 'tfidf':
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(train_text)
            X_test = vectorizer.transform(test_text)
        return X_train, X_test
