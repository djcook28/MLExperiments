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

# assumes a 2 column dataframe
def run_GaussianNB(DF, classes):
    x_train, x_test, class_train, class_test = (
        iris.create_train_test_split(x=DF, classification=classes, test_size=.65))

    # create gaussian naive bayes classifiers based on DF
    GNB = GaussianNB().fit(x_train, class_train)
    y_test_pred = GNB.predict(x_test)
    GNB_full = GaussianNB().fit(DF, classes)
    y_full_pred = GNB_full.predict(DF)

    # view the predictability accuracy results based on DF
    iris.print_classification_report(y_test=class_test, y_pred=y_test_pred)
    iris.print_classification_report(y_test=classes, y_pred=y_full_pred)

    # view the predictability accuracy results based on DF
    iris.decision_display(GNB, x_test, y_test_pred)
    iris.decision_display(GNB_full, DF, y_full_pred)

#run gaussian naive bayes against petal and sepal width to gauge classification fit
run_GaussianNB(pwswDF, iris.irisDF['Class'])

# Now doing the same with sepal length and petal width to be able to compare which categorization may be better
run_GaussianNB(slpwDF, iris.irisDF['Class'])

# the precision/recall of sepal length to petal width are better than sepal to petal width
# sepal length is a more accurate indicator of iris type

# it is important though to consider fact from data fit.  While this model fits well, does it make sense?
# looking at the decision boundaries they create oval shape boundaries which doesn't really make logical sense
# and we must consider here if a different model like a linear would be a better suited given what we know
# about how flower classification should actually behave