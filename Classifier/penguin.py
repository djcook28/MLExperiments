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

        self.kAccuracy_data = pd.DataFrame()

        self.bill_flip_df = self.penguin_df[['bill_length_mm', 'flipper_length_mm']]
        self.species = self.penguin_df['species']

        self.mass_billDepth_df = self.penguin_df[['body_mass_g', 'bill_depth_mm']]
        self.gender = self.penguin_df['sex']

    def execute_pairplot(self, classifier):
        super().execute_pairplot(df=self.penguin_df, classifier=classifier)

    def species_split_bill_flipper(self, test_size, random_state=42):
        bill_flip_train, bill_flip_test, species_train, species_test = (
            super().create_train_test_split(x=self.bill_flip_df, classification=self.species, test_size=test_size, random_state=random_state))
        return bill_flip_train, bill_flip_test, species_train, species_test

    def gender_split_mass_bill(self, test_size, random_state=42):
        mass_bill_train, mass_bill_test, gender_train, gender_test = (
            super().create_train_test_split(x=self.mass_billDepth_df, classification=self.gender, test_size=test_size, random_state=random_state))
        return mass_bill_train, mass_bill_test, gender_train, gender_test

    def kNeighbor_range(self, x_train, y_train, x_test, y_test, k_range):
        self.kAccuracy_data = super().kNeighbor_range(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, k_range=k_range)

    def species_kNeighbor(self, k):
        pred = super().kNeighbor(x=self.bill_flip_df, y=self.species, k=k)
        self.print_classification_report(y=self.species, y_pred=pred, target_names=self.species.unique())

    def gender_kNeighbor(self, k):
        pred = super().kNeighbor(x=self.mass_billDepth_df, y=self.gender, k=k)
        self.print_classification_report(y=self.gender, y_pred=pred, target_names=self.gender.unique())

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