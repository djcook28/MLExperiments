import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from classify import Classify

# load the penguin data set from the csv
penguin_data = pd.read_csv('../DataSets/penguins.csv')

# print some basic information about the dataset
###print(penguin_data.head())
###print(penguin_data.info())

# there us missing data and information regarding species, location, bill length and depth, flipper length, body mass
# and gender

#remove the missing data from the dataset
penguin_data.dropna(inplace=True)