from collections import Counter
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd


class DetectDrift:
    def __init__(self, test):
        self.test = test
        
    def get_proba(self, column1, column2):
        distribution_1 = Counter(column1)
        distribution_2 = Counter(column2)

        ensemble_categories = set(list(distribution_1.keys()) + list(distribution_2.keys()))

        probabilites_1 = np.array([distribution_1.get(cat, 0) / float(sum(distribution_1.values())) for cat in ensemble_categories])
        probabilites_2 = np.array([distribution_2.get(cat, 0) / float(sum(distribution_2.values())) for cat in ensemble_categories])

        return [probabilites_1, probabilites_2]
        
    def method_drift(self, dfs):
        benchmark_prob = []

        for df in dfs:    
            df_ref = df['Data'][0]
            df_new = df['Data'][1]
            
            for column in df_new.columns:
                probabilites       = self.get_proba(df_ref[column], df_new[column])
                drift_probability  = jensenshannon(probabilites[0], probabilites[1])
                is_drift           = self.detect_drift(drift_probability)
                
                if(df_new[column].dtype == np.float64 or df_new[column].dtype == np.int64):
                    category    = "numeric"
                else:
                    category    = "category"

                benchmark_prob.append({'Df': df['Stats'], 'column': column, 'type': category, 'js': round(drift_probability, 3), 'drift_js': is_drift})

        benchmark_df_pob = pd.DataFrame(benchmark_prob)
        return benchmark_df_pob
        
    def detect_drift(self, value):
        return (value > 0.1) if self.test == 'js' else \
            (value <= 0.05) if self.test == 'ks' else \
            (value > 0.1) if self.test == 'wd' else \
            (value < 0.05) if self.test == 'chi2' else \
            ValueError('Invalid test specified')