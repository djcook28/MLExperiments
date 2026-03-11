from iris import Iris

import warnings
warnings.filterwarnings('ignore')

iris = Iris()

#creates an iris pairplot to  view how the variables relate towards each other
# this provides insight into which variables are optimal to use for classification
# preferring variable relationships that are distinct rather than overlapping
# based off this sepal width vs petal width and sepal length vs petal width are good candidates
# these do have some overlap but not much which is actually preferred to help avoid overfitting
iris.execute_pairplot()

pwswDF = iris.irisDF['petal width (cm)'], ['sepal width (cm)']
slpwDF = iris.irisDF['sepal length (cm)'], ['petal width (cm)']

pwsw_test_class, pwsw_class_pred, pwsw_test_class_pred, pwsw_trainLR, pwsw_LR = iris.create_logreg(x=pwswDF, classification=iris.irisDF['Class'], test_size=.65)
slpw_test_class, slpw_class_pred, slpw_test_class_pred, slpw_trainLR, slpw_LR = iris.create_logreg(x=slpwDF, classification=iris.irisDF['Class'], test_size=.65)