import pandas as pd
from sklearn.datasets import load_iris
from classify import Classify
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

class Iris(Classify):
    irisDF = pd.DataFrame()

    def __init__(self):
        iris = load_iris()
        self.irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.irisDF['Class'] = iris.target
        self.sepals = self.irisDF[['sepal length (cm)', 'sepal width (cm)']]
        self.petals = self.irisDF[['petal length (cm)', 'petal width (cm)']]
        self.classification = self.irisDF['Class']
        self.petal_train_lr = LogisticRegression()
        self.petal_lr = LogisticRegression()
        self.sepal_train_lr = LogisticRegression()
        self.sepal_lr = LogisticRegression()

    def execute_pairplot(self):
        super().execute_pairplot(self.irisDF)

    def print_classification_report(self, y_test, y_pred):
        super().print_classification_report(y_test, y_pred, target_names=['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'])

    def create_sepal_logreg(self, test_size):
        test_class_pred, class_pred, class_test_set, self.sepal_train_lr, self.sepal_lr = (
            super().create_logreg(self.sepals, self.classification, test_size=test_size, random_state=42))

        return test_class_pred, class_pred, class_test_set

    def create_petal_logreg(self, test_size):
        test_class_pred, class_pred, class_test_set, self.petal_train_lr, self.petal_lr =(
            super().create_logreg(self.petals, self.classification, test_size=test_size, random_state=42))

        return test_class_pred, class_pred, class_test_set

    def petal_decision_display(self):
        graph = super().decision_display(self.petal_lr, self.petals, self.petals.columns[0], self.petals.columns[1])
        graph.ax_.scatter(self.petals.iloc[:,0], self.petals.iloc[:,1], c=self.classification)
        plt.show()

    def sepal_decision_display(self):
        graph = super().decision_display(self.sepal_lr, self.sepals, self.sepals.columns[0], self.sepals.columns[1])
        graph.ax_.scatter(self.sepals.iloc[:, 0], self.sepals.iloc[:, 1], c=self.classification)
        plt.show()