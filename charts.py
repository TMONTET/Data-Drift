import matplotlib.pyplot as plt
import seaborn as sns

class Graphics:
    def plot_hist(self, df_ref, df_new, column_name):
        # Tracer l'histogramme de chaque colonne des deux dataframes
        sns.histplot(df_ref[column_name], color="blue", multiple="stack", kde=True, label='Ref')
        sns.histplot(df_new[column_name], color="red", multiple="stack", kde=True, label='New')
        # Ajouter une légende
        plt.legend()
        # Afficher le graphique
        plt.show()
        
    def plot_graph(self, data):
        # Tracer un graphe de points en fonction de l'index
        plt.plot(data.index, data)

        # Ajouter une légende pour les axes
        plt.xlabel('Index')
        plt.ylabel('Valeur')

        # Afficher le graphe
        plt.show()