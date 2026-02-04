import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

class Classify:
    def execute_pairplot(self, df):
        sns.set_style('darkgrid')

        graph = sns.pairplot(df, kind='scatter', hue='Class', palette='bright')
        plt.show()

    def print_classification_report(self, y_test, y_pred, target_names):
        print(classification_report(y_test, y_pred, target_names=target_names))