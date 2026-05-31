from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.naive_bayes import GaussianNB

from iris import Iris

import warnings
warnings.filterwarnings('ignore')

iris = Iris()

#creates an iris pairplot to  view how the variables relate towards each other
# this provides insight into which variables are optimal to use for classification
# preferring variable relationships that are distinct rather than overlapping
# based off this sepal width vs petal width and sepal length vs petal width are good candidates
# these do have some overlap but not much which is actually preferred to help avoid overfitting
###iris.execute_pairplot()

pwswDF = iris.irisDF[['petal width (cm)', 'sepal width (cm)']]
slpwDF = iris.irisDF[['sepal length (cm)', 'petal width (cm)']]

pwsw_x_train, pwsw_x_test, pwsw_class_train, pwsw_class_test = (
    iris.create_train_test_split(x=pwswDF,classification=iris.irisDF['Class'], test_size=.65))
slpw_x_train, slpw_x_test, slpw_class_train, slpw_class_test = (
    iris.create_train_test_split(x=slpwDF, classification=iris.irisDF['Class'], test_size=.65))

pwswGNB = GaussianNB().fit(pwsw_x_train, pwsw_class_train)
pwsw_y_test_pred = pwswGNB.predict(pwsw_x_test)
pwswGNB_full = GaussianNB().fit(pwswDF, iris.classification)
pwsw_y_full_pred = pwswGNB_full.predict(pwswDF)

iris.print_classification_report(y_test=pwsw_class_test, y_pred=pwsw_y_test_pred)
iris.print_classification_report(y_test=iris.classification, y_pred=pwsw_y_full_pred)

iris.decision_display(pwswGNB, pwsw_x_test, pwsw_y_test_pred)
iris.decision_display(pwswGNB_full, pwswDF, pwsw_y_full_pred)

slpwGNB = GaussianNB().fit(slpw_x_train,slpw_class_train)
slpw_y_pred = slpwGNB.predict(slpw_x_test)
iris.print_classification_report(y_test=slpw_class_test, y_pred=slpw_y_pred)