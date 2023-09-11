# TODO: add fit test before prediction
import numpy as np


class NaiveBayes:
    def __init__(self):
        ...

    def fit(self, X: np.array, y: np.array):
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
        freq_matrix = np.zeros((len(self.classes), vocab_size))
        for i, class_ in enumerate(self.classes):
            class_mask = (y == class_).flatten()

            n = X[class_mask].sum()  # number of words that belong to class
            n_k = X[class_mask].sum(axis=0)

            # Generate words freq by formula
            # P(w_k | class) = (n_k + 1) / (n + |vocabulary|)
            self.words_freqs = (n_k + 1) / (n + vocab_size)  # smoothing

            freq_matrix[i] = self.words_freqs

        return self

    def predict(self, X: np.array):
        """
        Learn dataset vocabulary.

        Parameters
        ----------
        X : iterable
            Input samples.

        Returns
        -------
        C : np.array
            Predict target values for X.
        """
        return np.apply_along_axis(self._estimate_class, 1, X)
        
    def _estimate_class(self, sample: np.array):
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

        max_estimated_prob = 0
        predicted_class = None
        for i, class_ in enumerate(self.classes):
            class_log_prob = np.log(self.classes_freqs[i])
            sample_log_prob = np.sum(sample * self.words_freqs[i])
            estimated_prob = class_log_prob + sample_log_prob

            if estimated_prob > max_estimated_prob:
                max_estimated_prob = estimated_prob
                predicted_class = class_

        return predicted_class
