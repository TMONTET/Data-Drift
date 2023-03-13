from scipy.spatial.distance import jensenshannon
import numpy as np
import random
import pandas as pd
from detect_drift import DetectDrift


class AnalyseDrift:
    def __init__(self, test):
        self.test      = test
        self.detection = DetectDrift("js")
    
    def split_data(self, dfs):
        test_ref = dfs[(len(dfs)-1)]['Data'][0]
        test_new = dfs[(len(dfs)-1)]['Data'][1]

        # Division en 4 parties égales
        df_list_ref = np.array_split(test_ref, 1000)
        df_list_new = np.array_split(test_new, 1000)

        for test in df_list_ref[750:1000]:
            df_list_new.append(test)

        for test in df_list_ref[500:750]:
            df_list_new.insert(0, test)

        del df_list_ref[500:1000]
        df_list_ref = pd.concat(df_list_ref)
        
        return df_list_ref, df_list_new

    def method_timeline(self, splited_data_ref, splited_data_new):
        js_time_cat = []
        js_time_num = []

        for batch in splited_data_new:
            for column in batch.columns:
                
                probs       = self.detection.get_proba(splited_data_ref[column], batch[column])
                drift_probs = jensenshannon(probs[0], probs[1])
                
                if(batch[column].dtype == np.float64 or batch[column].dtype == np.int64):
                    js_time_num.append(drift_probs)
                else:
                    js_time_cat.append(drift_probs)
                

        js_time_cat_df = pd.DataFrame(js_time_cat)
        js_time_num_df = pd.DataFrame(js_time_num)
        return js_time_cat_df, js_time_num_df
    
    
    def average_method(self, dfs):
        ref = dfs[(len(dfs)-1)]['Data'][0]

        col_cat = ref.loc[:, 'Category']
        col_num = ref.loc[:, 'Number']


        # Division en 4 parties égales
        list_cat_moy = np.array_split(col_cat, 1000)
        list_num_moy = np.array_split(col_num, 1000)
        
        # Créer une liste de tous les nombres de 1 à 100
        liste_1_a_100 = list(range(0, 999))

        # Créer une liste de toutes les combinaisons uniques de deux chiffres aléatoires
        list_combinations = [(a, b) for a in random.sample(liste_1_a_100, 100) for b in random.sample(liste_1_a_100, 100) if a != b]
        
        avg_js_cat = []
        avg_js_num = []

        for batch in list_combinations:
            probs_cat       = self.detection.get_proba(list_cat_moy[batch[0]], list_cat_moy[batch[1]])
            probs_num       = self.detection.get_proba(list_num_moy[batch[0]], list_num_moy[batch[1]])
            drift_probs_cat = jensenshannon(probs_cat[0], probs_cat[1])
            drift_probs_num = jensenshannon(probs_num[0], probs_num[1])
            avg_js_cat.append(drift_probs_cat)
            avg_js_num.append(drift_probs_num)
        
        moyenne_cat = np.mean(avg_js_cat)
        moyenne_num = np.mean(avg_js_num)
        
        return moyenne_cat, moyenne_num