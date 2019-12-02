import numpy as numpy
import sys


class DTClassifier:
    def __init__(self, max_depth, min_leaf_samples):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples

    @staticmethod
    def