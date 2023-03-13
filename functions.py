import numpy as np
from collections import Counter

class DriftFunctions:
    def __init__(self, test):
        self.test = test
    
    def get_proba(self, column1, column2):
        distribution_1 = Counter(column1)
        distribution_2 = Counter(column2)

        ensemble_categories = set(list(distribution_1.keys()) + list(distribution_2.keys()))

        probabilites_1 = np.array([distribution_1.get(cat, 0) / float(sum(distribution_1.values())) for cat in ensemble_categories])
        probabilites_2 = np.array([distribution_2.get(cat, 0) / float(sum(distribution_2.values())) for cat in ensemble_categories])

        return [probabilites_1, probabilites_2]
    
    def detect_drift(self, value):
        return (value > 0.1) if self.test == 'js' else \
            (value <= 0.05) if self.test == 'ks' else \
            (value > 0.1) if self.test == 'wd' else \
            (value < 0.05) if self.test == 'chi2' else \
            ValueError('Invalid test specified')