# TODO: add fit test before prediction
from typing import Any
from typing_extensions import Self
import numpy as np


class NaiveBayes:
    def __init__(self):
        ...

    def fit(self, X: np.array, y: np.array) -> Self:
        """
        Fit MultiNomial Naive Bayes.

        Parameters
        ----------
        X : np.array
            Training vectors of shape (n_samples, n_features).
        y : np.array
            Target values.

        Returns
        -------
        self : object
            Fitted model.
        """
        
        # Get frequenices for target classes
        self.classes, self.classes_counts = np.unique(y, return_counts=True)
        self.classes_freqs = self.classes_counts / sum(self.classes_counts)
        
        vocab_size = X.shape[1]
        self.freq_matrix = np.zeros((len(self.classes), vocab_size))
        for i, class_ in enumerate(self.classes):
            class_mask = (y == class_).flatten()

            n = X[class_mask].sum()  # number of words that belong to class
            n_k = X[class_mask].sum(axis=0)

            # Generate words freq by formula
            # P(w_k | class) = (n_k + 1) / (n + |vocabulary|)
            words_freqs = (n_k + 1) / (n + vocab_size)  # smoothing
            self.freq_matrix[i] = words_freqs

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Learn dataset vocabulary.

        Parameters
        ----------
        X : iterable
            Input samples.

        Returns
        -------
        y_pred : np.array
            Predict target values for X.
        """
        y_pred = np.zeros((len(X), 1), dtype=self.classes.dtype)
        for i, row in enumerate(X):
            y_pred[i] = self._estimate_class(row)
        return y_pred
        
    def _estimate_class(self, sample: np.array) -> Any:
        """
        Estimate probability of a sample belonging to class
        P(class | w_1, w_2, ..., w_n) ~ ln(P(class)) + sum_i ln(P(w_i | class))
        Get class with maximum estimated probability.

        Parameters
        ----------
        sample : np.array
            Input sample.

        Returns
        -------
        predicted_class : Any
            Class with maximum estimated probability.
        """

        max_estimated_prob = -float('inf')
        predicted_class = None
        for i, class_ in enumerate(self.classes):
            class_log_prob = np.log(self.classes_freqs[i])
            sample_log_prob = np.sum(sample * np.log(self.freq_matrix[i]))
            estimated_prob = class_log_prob + sample_log_prob

            if estimated_prob > max_estimated_prob:
                max_estimated_prob = estimated_prob
                predicted_class = class_

        return predicted_class


if __name__ == '__main__':
    import sys
    sys.path.append('../')

    from feature_extraction.main import BagOfWords

    dataset = [
        'You are a bad boy',
        'You are piece of shit',
        'Hey, you are nice'
    ]
    y_train = np.array(['toxic', 'toxic', 'not toxic'])
    sample = ['you are bad', 'you are nice', 'you are nice']

    bow = BagOfWords()
    X_train = bow.fit_transform(dataset)
    X_test = bow.transform(sample)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print(y_pred)



    


