from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from penguin import Penguin

penguin = Penguin()

def plotKAccuracy(classification):
    fig, ax = plt.subplots()
    penguin.kAccuracy_data.plot(kind='line', x='K', y='Trained_accuracy', ax=ax, c='red', label='trained')
    penguin.kAccuracy_data.plot(kind='line', x='K', y='Tested_accuracy', ax=ax, c='blue', label='tested')
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel(classification)
    plt.show()

# run a k neighbor classifier on penguin species using bill and flipper lengths
x_train, x_test, y_train, y_test = penguin.species_split_bill_flipper(test_size=0.4)
penguin.kNeighbor_range(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, k_range=40)
print(penguin.kAccuracy_data)
plotKAccuracy(classification='Species')

# Based on the results a K of 1 for species seems optimal though it may result in overfitting, k=2 or 3 may be
#more advisable as there could be greater variability in a larger dataset.
penguin.species_kNeighbor(k=2)

# run a k neighbor classifier on penguin gender using bill depth and body mass
x_train, x_test, y_train, y_test = penguin.gender_split_mass_bill(test_size=0.4)
penguin.kNeighbor_range(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, k_range=40)
print(penguin.kAccuracy_data)
plotKAccuracy(classification='Gender')

# Based on the results a K of 6 for gender seems optimal.  While K = 6 has a lower trained
#accuracy it does have a higher test accuracy as opposed ti K = 1 for gender
penguin.gender_kNeighbor(k=6)