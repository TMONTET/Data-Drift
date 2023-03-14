from collections import Counter
import time
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
from river import drift
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

    def detect_drift(self, value):
        return (value > 0.1) if self.test == 'js' else \
            (value < 0.05) if self.test == 'ks' else \
            (value > 0.1) if self.test == 'wd' else \
            (value < 0.05) if self.test == 'chi2' else \
            ValueError('Invalid test specified')
    
    def switch_method(self, ref, new):
        if self.test == "ks":
            ks_stat, p_value = ks_2samp(ref, new)
            return p_value
        if self.test == "chi2":
            chi2, p_value, dof, expected = chi2_contingency([ref,  new])
            return p_value

    def method_drift(self, dfs):
        benchmark_prob_detail = []
        benchmark_prob        = []

        for dataset in dfs:
            df_ref = dataset['Data'][0]
            df_new = dataset['Data'][1]
            all_correct = True
            total_time  = 0
            
            for column in df_new.columns:
                correct_column     = True
                probabilites       = self.get_proba(df_ref[column], df_new[column])
                
                switcher = {
                    'js': jensenshannon(probabilites[0], probabilites[1]),
                    'wd': wasserstein_distance(probabilites[0], probabilites[1]),
                    'ks': self.switch_method(probabilites[0], probabilites[1]),
                    'chi2': self.switch_method(probabilites[0], probabilites[1]),
                    'default': jensenshannon(probabilites[0], probabilites[1])
                }
                
                start_time = time.time()
                drift_probability  = switcher.get(self.test, jensenshannon(probabilites[0], probabilites[1]))
                elapsed_time = time.time() - start_time
                total_time  += elapsed_time
                
                is_drift           = self.detect_drift(drift_probability)
                
                
                if(df_new[column].dtype == np.float64 or df_new[column].dtype == np.int64):
                    category    = "numeric"
                    
                    if is_drift == True:
                        correct_column = False
                        all_correct    = False
                    
                else:
                    category    = "category"
                    
                    if is_drift == False:
                        correct_column = False
                        all_correct    = False

                benchmark_prob_detail.append({'Df': dataset['Stats'], 'column': column, 'type': category, self.test: round(drift_probability, 3), 'drift_detected': is_drift, 'correct': correct_column, 'time': elapsed_time})
        
            benchmark_prob.append({'Num_Cat': dataset['Num_Cat'], 'Num_Number': dataset['Num_Number'], 'Right': all_correct, 'Time': total_time})
        
        df_benchmark_pob        = pd.DataFrame(benchmark_prob)
        df_benchmark_pob_detail = pd.DataFrame(benchmark_prob_detail)
        return df_benchmark_pob_detail, df_benchmark_pob

    def adwin(self, df):
        
        adwin = drift.ADWIN()

        for i, val in enumerate(df.iterrows()):
            index, data = val
            _ = adwin.update(data[0])
            if adwin.drift_detected:
                print(f"Change detected at index {i}, input value: {val}")
        
    def final_benchmark(self, all_benchmark):
        final_benchmark = []

        for i in range(len(all_benchmark)):
            result    = all_benchmark[i]["Result"][1]
            method    = all_benchmark[i]["Test"]
            counts    = result["Right"].value_counts()
            mean_time = result['Time'].mean()

            try:
                ratio  = counts[True] / counts.sum()
            except:
                ratio  = 0
                
            final_benchmark.append({'Method': method, 'Correct_Ratio': round(ratio, 3), 'Avg_Time': mean_time})

        df_final_benchmark = pd.DataFrame(final_benchmark)
        return df_final_benchmark