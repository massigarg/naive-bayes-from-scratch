import numpy as np
import pandas as pd
from sklearn import datasets, naive_bayes

import math
import scipy.stats as stats


np.random.seed(1345)

# Load the wine dataset (description here http://scikit-learn.org/stable/datasets/index.html#diabetes-dataset)
wine = datasets.load_wine()
data = wine.data.copy()
target = wine.target.copy()

# Split the data into training/testing sets
total_samples = wine.target.shape[0]
exclude = round(total_samples/3)
indices = np.arange(0, total_samples)
np.random.shuffle(indices)

idx_train = indices[:-exclude]
idx_test = indices[-exclude:]

assert not np.intersect1d(idx_test, idx_train).size

X_train = data[idx_train]
X_test = data[idx_test]

# Split the targets into training/testing sets
y_train = target[idx_train]
y_test = target[idx_test]


class myGaussianNB:
    """
    Naive Bayes for continous variables -> Gaussian Naive Bayes

    Bayes Theorem
    P(y|X) = P(X|y) * P(y) / P(X)
    """

    def __init__(self):
        # initialise the attributes of this class
        self.classes = []
        self.features_number = 0

        # self.class_prior = dict()
        # self.class_mean = dict()
        # self.class_std = dict()
        self.class_likelihood = dict()
        self.posteriors = []
        self.predictions = []

    def class_mean(self, features, target):
        """
        Compute the mean for each feature

        Args:
            features (ndarray): 2D features array
            target (ndarray): 1D target array
        Returns:
            ndarray: mean array for each feature
        """
        self.class_mean = {}

        for c in self.classes:
            self.class_mean[c] = features[target == c].mean(
                0)  # np.mean() on 0 axis

        return self.class_mean

    def class_std(self, features, target, ddof=1):
        """
        Compute corrected sample standard deviation.
        To copute uncorrected sample standard deviation change ddof to 0

        Args:
            features (ndarray): 2D features array
            target (ndarray): 1D target array
            ddof (int): Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.

        Returns:
            ndarray: array of the standard deviation for each feature
        """

        self.class_std = {}

        for c in self.classes:
            self.class_std[c] = features[target == c].std(
                0, ddof=ddof)  # np.std() on 0 axis

        return self.class_std

    def class_prior(self, target):
        """
        In Bayesian statistical inference, a prior probability distribution, often simply called the prior,
        of an uncertain quantity is the probability distribution that would express one's beliefs about
        this quantity before some evidence is taken into account. (Wikipedia, 2021)

        Thus we will evaluate the prior as the propability distribuition of encountering either class 0,1 or 2.

        Args:
            target (ndarray): 1D target array

        Returns:
            ndarray: array of length # of calsses 
        """
        self.class_prior = {}

        for c in self.classes:
            self.class_prior[c] = np.sum(target == c) / len(target)

        return self.class_prior

    def fit(self, features, target):
        self.classes = np.unique(target.astype(int))
        self.features_number = features.shape[1]

        self.class_mean(features, target)
        self.class_std(features, target)
        self.class_prior(target)

    def predict(self, X_test):
        # 1. evaluate (log) likelihoods of test data for each class
        for c in self.classes:

            # there will be multiple gaussians that need to be combined using the naive assumption
            likelihood = 1
            for obs in np.arange(0, self.features_number).astype(int):
                likelihood = likelihood * \
                    stats.norm.pdf(
                        X_test[:, obs], self.class_mean[c][obs], self.class_std[c][obs])
                #likelihood = likelihood * stats.norm.pdf(X_test[:,obs], self.class_mean[c][obs], self.class_std[c][obs])
            self.class_likelihood[c] = likelihood

            # 2. approximate the posterior using P(X|Y)P(Y)
            self.posteriors.append(
                self.class_prior[c] * self.class_likelihood[c])

        # 3. take the maximum posterior probability as our final class
        self.predictions = np.argmax(self.posteriors, 0)

        return self.predictions
