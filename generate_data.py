import math
import random
import pandas as pd

class RandomDataGenerator:
    # Génération de données aléatoires
    def generate_random_data(self, n, categories, weights):
        for i in range(n):
            category = random.choices(categories, weights=weights)[0]
            # number   = random.randint(1, 20)
            number = random.choices([10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], weights=[9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])[0]

            yield {'Category': category, 'Number': number}

    # Création des deux dataframes avec le nombre de donnée et de catégorie différente que je veux
    def get_data(self, n_data, n_categories):
        categories          = ['fruit', 'légume', 'viande', 'poisson', 'intention','disk','player','population','oven','student','movie','agreement','procedure','actor', 'union', 'error', 'employee', 'security', 'region', 'user', 'cell', 'internet', 'wife', 'clothes']
        selected_categories = random.sample(categories, n_categories)

        if n_categories == 2:
            position_ref = 0
            position_new = 1
        else:
            position_ref = random.randrange(0, (n_categories//2))
            position_new = random.randrange((n_categories//2)+1, n_categories)

        weights_ref = [random.choice([0.1, 0.2, 0.3]) if i != position_ref else 0.9 for i in range(n_categories)]
        weights_new = [random.choice([0.1, 0.2, 0.3]) if i != position_new else 0.9 for i in range(n_categories)]

        random_data_ref = self.generate_random_data(n_data, selected_categories, weights_ref)
        random_data_new = self.generate_random_data(n_data, selected_categories, weights_new)

        df_ref = pd.DataFrame(random_data_ref)
        df_new = pd.DataFrame(random_data_new)
        
        return [df_ref, df_new]

    # Création de la liste des rangs de données et de colonnes
    def number_data(self, data_range_min, data_range_max, range_cat_min, range_cat_max):
        data_list = []
        data      = data_range_min
        while data <= data_range_max:
            data_list.append(data)
            data *= 10
            
        col_list = []
        col_list.append(range_cat_min)
        col_start = math.ceil((range_cat_min+1)/10)*10
        for i in range(col_start, range_cat_max+1, 10):
            col_list.append(i)
        
        # Generate Data
        dfs = []
        for n_data in data_list:
            for n_categories in col_list:
                df_ref, df_new = self.get_data(n_data, n_categories)
                df_name = str(n_data) + "_data_&_" + str(n_categories) + "_cat"
                dfs.append({'Stats': df_name, "Num_Cat": n_categories, "Num_Number": n_data,'Data':(df_ref, df_new)})
        
        return dfs