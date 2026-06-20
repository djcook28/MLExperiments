import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class Classify:
    # takes in a data frame and generates a pair plot to view the relationship of data frame variables
    def execute_pairplot(self, df, classifier):
        sns.set_style('darkgrid')

        graph = sns.pairplot(df, kind='scatter', hue=classifier, palette='bright')
        plt.show()

    def print_classification_report(self, y, y_pred, target_names):
        print(classification_report(y, y_pred, target_names=target_names))

    def create_train_test_split(self, x, classification, test_size, random_state=42):
        x_train, x_test, class_train, class_test = train_test_split(x, classification,
                                                                        test_size=test_size, random_state=random_state)
        return x_train, x_test, class_train, class_test

    def create_logreg(self, x, classification, test_size, random_state=42):
        x_train, x_test, class_train, class_test = self.create_train_test_split(x, classification,
                                                                            test_size=test_size, random_state=random_state)
        train_lr = LogisticRegression().fit(x_train, class_train)
        test_class_pred = train_lr.predict(x_test)

        lr = LogisticRegression().fit(x, classification)
        class_pred = lr.predict(x)
        return test_class_pred, class_pred, class_test, train_lr, lr

    def decision_display(self, lr, x, x_label, y_label, cmap):
        graph = DecisionBoundaryDisplay.from_estimator(lr, x, response='predict', xlabel=x_label, ylabel=y_label, cmap=cmap)
        return graph

    def kNeighbor_range(self, x_train, y_train, x_test, y_test, k_range):

        kAccuracy_data = pd.DataFrame(columns=['K','Trained_accuracy', 'Tested_accuracy'])

        for k in range(1, k_range):
            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            y_train_pred = classifier.predict(x_train)
            Tr_accuracy = accuracy_score(y_train, y_train_pred)
            Te_accuracy = accuracy_score(y_test, y_pred)
            accuracy_values = pd.DataFrame.from_dict(
                {'K': [k], 'Trained_accuracy': [Tr_accuracy], 'Tested_accuracy': [Te_accuracy]})
            kAccuracy_data = pd.concat([kAccuracy_data, accuracy_values], ignore_index=True)

        return kAccuracy_data

    def kNeighbor(self, x, y, k):
        pred = KNeighborsClassifier(n_neighbors=k).fit(x, y)
        return pred.predict(x)