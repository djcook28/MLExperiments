import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iris import Iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay

import warnings
warnings.filterwarnings('ignore')

iris = Iris()

print(iris.irisDF)

#creating an iris pairplot to be able to view how the variables relate towards each other graphically in order
# to determine which we will want to use for different demonstrations of classification methods and issues
# which variables already show class distinictions vs which show overlap
iris.execute_pairplot()

def create_logreg(x, y, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    logreg = LogisticRegression().fit(x_train, y_train)
    logreg2 = LogisticRegression().fit(x, y)

    y_pred = logreg.predict(x_test)
    y_pred2 = logreg2.predict(x)
    return y_pred, y_pred2, y_test

sepal_x = iris.irisDF[['sepal length (cm)', 'sepal width (cm)']]
petal_x = iris.irisDF[['petal length (cm)', 'petal width (cm)']]
class_y = iris.irisDF['Class']


#this next step splits the x,y datas into a train and test data set.  test size .65 puts 65% of the data in the test
# leaving the remaining 35% for the train data set
sepal_test_pred, sepal_pred, sepal_test = create_logreg(sepal_x, class_y, .65)
petal_test_pred, petal_pred, petal_test = create_logreg(petal_x, class_y, .65)

#in classification there are a couple important results,
# 1 is Precision which defines out of how many things identified as being that class, actually are that class.
# 2 if recall which says out of all the items that actually are that class, how many did the model classify as that class
# 3 F1 which is recall*precision over (recall+precision) which identifies how good the model is at meeting both
iris.print_classification_report(sepal_test, sepal_test_pred)
iris.print_classification_report(class_y, sepal_pred)
iris.print_classification_report(petal_test, petal_test_pred)
iris.print_classification_report(class_y, petal_pred)