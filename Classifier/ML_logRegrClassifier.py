from iris import Iris

import warnings
warnings.filterwarnings('ignore')

iris = Iris()

print(iris.irisDF)

#creating an iris pairplot to be able to view how the variables relate towards each other graphically in order
# to determine which we will want to use for different demonstrations of classification methods and issues
# which variables already show class distinictions vs which show overlap
iris.execute_pairplot()

#this next step splits the x,y datas into a train and test data set.  test size .65 puts 65% of the data in the test
# leaving the remaining 35% for the train data set
test_class_pred_sepal, class_pred_sepal, sepal_test_set = iris.create_sepal_logreg(test_size=.65)
test_class_pred_petal, class_pred_petal, petal_test_set = iris.create_petal_logreg(test_size=.65)

#in classification there are a couple important results,
# 1 is Precision which defines out of how many things identified as being that class, actually are that class.
# 2 is recall which says out of all the items that actually are that class, how many did the model classify as that class
# 3 F1 which is recall*precision over (recall+precision) which identifies how good the model is at meeting both
iris.print_classification_report(sepal_test_set, test_class_pred_sepal)
iris.print_classification_report(iris.classification, class_pred_sepal)
iris.print_classification_report(petal_test_set, test_class_pred_petal)
iris.print_classification_report(iris.classification, class_pred_petal)

#this will visually display the boundaries between classifications as decided by the LR model
iris.petal_decision_display()
iris.sepal_decision_display()