# TODO: add working with missing values
# TODO: if asked vocabulary return just dict not defaultdict
# TODO: run linters


from collections import defaultdict
from collections.abc import Callable, Iterable

import numpy as np


class BagOfWords:
    """
    Convert a collection of text documents to a bag of words model.

    Parameters
    ----------
    tokenizer : callable, default=lambda x: x.split()
        Word tokenization function

    count_method : {'counts', 'presence'}, default='counts'
        Option 'counts' - count every occurence of token in sentence
        Option 'presence' - count only presence of token in sentence
    """
    def __init__(
            self,
            tokenizer: Callable[[str], list[str]] = lambda x: x.split(),
            count_method: str = 'counts') -> None:
        self.tokenizer = tokenizer
        self.count_method = count_method

    def _create_vocabulary(self, dataset: Iterable[str]) -> np.array:
        # For each unseen key assign length of the vocabulary as value
        self.vocabulary = defaultdict()
        self.vocabulary.default_factory = self.vocabulary.__len__
        
        # Create bag of words representation of dataset
        samples_indices = []
        samples_counts = []
        for document in dataset:
            idx_counter = {}
            for token in self.tokenizer(document):
                # Add token to vocabulary and get token index
                token_idx = self.vocabulary[token]

                if self.count_method == 'counts':
                    if token_idx not in idx_counter:
                        idx_counter[token_idx] = 0
                    idx_counter[token_idx] += 1
                elif self.count_method == 'presence':
                    if token_idx not in idx_counter:
                        idx_counter[token_idx] = 1

            samples_indices.append(list(idx_counter.keys()))
            samples_counts.append(list(idx_counter.values()))

        matrix = np.zeros((len(dataset), len(self.vocabulary)))

        for i in range(len(dataset)):
            matrix[i][samples_indices[i]] = samples_counts[i]

        return matrix

    def fit_transform(self, dataset: Iterable[str]) -> np.array:
        """
        Create vocabulary dictionary and return document term matrix.

        Parameters
        ----------
        dataset : iterable
            An iterable which generates str.

        Returns
        -------
        C : array of shape (n_samples, n_features)
            Document term matrix.
        """

        return self._create_vocabulary(dataset)

    def fit(self, dataset: Iterable[str]):
        """
        Learn dataset vocabulary.

        Parameters
        ----------
        dataset : iterable
            An iterable which generates str.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """

        self._create_vocabulary(dataset)

        return self

    def transform(self, dataset: Iterable[str]) -> np.array:
        """
        Transform dataset using fitted vocabulary.

        Parameters
        ----------
        dataset : iterable
            An iterable which generates str.

        Returns
        -------
        C : array of shape (n_samples, n_features)
            Document term matrix.
        """
        samples_indices = []
        samples_counts = []
        for document in dataset:
            idx_counter = {}
            for token in self.tokenizer(document):
                if token in self.vocabulary:
                    token_idx = self.vocabulary[token]
                    if self.count_method == 'counts':
                        if token_idx not in idx_counter:
                            idx_counter[token_idx] = 0
                        idx_counter[token_idx] += 1
                    elif self.count_method == 'presence':
                        if token_idx not in idx_counter:
                            idx_counter[token_idx] = 1

            samples_indices.append(list(idx_counter.keys()))
            samples_counts.append(list(idx_counter.values()))

        matrix = np.zeros((len(dataset), len(self.vocabulary)))

        for i in range(len(dataset)):
            matrix[i][samples_indices[i]] = samples_counts[i]

        return matrix


if __name__ == '__main__':
    dataset_train = [
        'auf mashina auf jopa',
        'wow hello',
        'wow auf fara'
    ]

    dataset_test = [
        'wow privet',
        'kak ti auf'
    ]
    
    vectorizer = BagOfWords()
    bow = vectorizer.fit_transform(dataset_train)

    print(vectorizer.vocabulary)
    print(bow)

    print(vectorizer.transform(dataset_test))


