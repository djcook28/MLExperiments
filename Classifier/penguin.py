import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from classify import Classify

class Penguin(Classify):
    def __init__(self):
        # load the penguin data set from the csv
        self.penguin_df = pd.read_csv('../DataSets/penguins.csv')
        self.penguin_df.dropna(inplace=True)

        self.bill_flip_df = self.penguin_df[['bill_length_mm', 'flipper_length_mm']]
        self.species = self.penguin_df['species']

    def execute_pairplot(self, classifier):
        super().execute_pairplot(df=self.penguin_df, classifier=classifier)

    def species_split_bill_flipper(self, test_size, random_state=42):
        bill_flip_train, bill_flip_test, species_train, species_test = (
            super().create_train_test_split(x=self.bill_flip_df, classification=self.species, test_size=test_size, random_state=random_state))
        return bill_flip_train, bill_flip_test, species_train, species_test

if __name__ == '__main__':
    penguin = Penguin()
    # print some basic information about the dataset
    print(penguin.penguin_df.head())
    print(penguin.penguin_df.info())

    # there is missing data and information regarding species, location, bill length and depth, flipper length, body mass
    # and gender

    # now using the pair plot to find variable combinations that provide clearly defined class boundaries
    # for potential use cases like predicting species, gender or island.
    penguin.execute_pairplot('species')
    penguin.execute_pairplot('sex')
    penguin.execute_pairplot('island')
    # the only readily usable classification is species.  Gender and Island have significant overlaps that would
    # require additional data prep