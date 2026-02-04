import pandas as pd
from sklearn.datasets import load_iris
from classify import Classify

class Iris(Classify):
    irisDF = pd.DataFrame()

    def __init__(self):
        iris = load_iris()
        self.irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.irisDF['Class'] = iris.target

    def execute_pairplot(self):
        super().execute_pairplot(self.irisDF)

    def print_classification_report(self, y_test, y_pred):
        super().print_classification_report(y_test, y_pred, target_names=['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'])