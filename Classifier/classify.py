import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

class Classify:
    def execute_pairplot(self, df):
        sns.set_style('darkgrid')

        graph = sns.pairplot(df, kind='scatter', hue='Class', palette='bright')
        plt.show()

    def print_classification_report(self, y_test, y_pred, target_names):
        print(classification_report(y_test, y_pred, target_names=target_names))

    def create_logreg(self, x, classification, test_size, random_state=42):
        x_train, x_test, class_train, class_test_set = train_test_split(x, classification,
                                                                            test_size=test_size, random_state=random_state)
        train_lr = LogisticRegression().fit(x_train, class_train)
        test_class_pred = train_lr.predict(x_test)

        lr = LogisticRegression().fit(x, classification)
        class_pred = lr.predict(x)
        return test_class_pred, class_pred, class_test_set, train_lr, lr

    def decision_display(self, lr, x, x_label, y_label, cmap):
        graph = DecisionBoundaryDisplay.from_estimator(lr, x, response='predict', xlabel=x_label, ylabel=y_label, cmap=cmap)
        return graph