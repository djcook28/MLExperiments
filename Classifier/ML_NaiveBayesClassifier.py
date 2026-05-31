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

# create gaussian naive bayes classifiers based on petal and sepal width
pwswGNB = GaussianNB().fit(pwsw_x_train, pwsw_class_train)
pwsw_y_test_pred = pwswGNB.predict(pwsw_x_test)
pwswGNB_full = GaussianNB().fit(pwswDF, iris.classification)
pwsw_y_full_pred = pwswGNB_full.predict(pwswDF)

# view the predictability accuracy results based on petal and sepal width
iris.print_classification_report(y_test=pwsw_class_test, y_pred=pwsw_y_test_pred)
iris.print_classification_report(y_test=iris.classification, y_pred=pwsw_y_full_pred)

# view the decision boundaries of the test and full set based on the petal and sepal width NB
iris.decision_display(pwswGNB, pwsw_x_test, pwsw_y_test_pred)
iris.decision_display(pwswGNB_full, pwswDF, pwsw_y_full_pred)

# Now doing the same with sepal length and petal width to be able to compare which categorization may be better
slpwGNB = GaussianNB().fit(slpw_x_train,slpw_class_train)
slpw_y_test_pred = slpwGNB.predict(slpw_x_test)
slpwGNB_full = GaussianNB().fit(slpwDF, iris.classification)
slpw_y_full_pred = slpwGNB_full.predict(slpwDF)

iris.print_classification_report(y_test=slpw_class_test, y_pred=slpw_y_test_pred)
iris.print_classification_report(y_test=iris.classification, y_pred=slpw_y_full_pred)

iris.decision_display(slpwGNB, slpw_x_test, slpw_y_test_pred)
iris.decision_display(slpwGNB_full, slpwDF, slpw_y_full_pred)

# the precision/recall of sepal length to petal width are better than sepal to petal width
# sepal length is a more accurate indicator of iris type