from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from penguin import Penguin

penguin = Penguin()
x_train, x_test, y_train, y_test = penguin.species_split_bill_flipper(test_size=0.4)

accuracy_data = pd.DataFrame(columns=['K','Trained_accuracy', 'Tested_accuracy'])

for k in range(1, 40):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_train_pred = classifier.predict(x_train)
    Tr_accuracy = accuracy_score(y_train, y_train_pred)
    Te_accuracy = accuracy_score(y_test, y_pred)
    accuracy_values = pd.DataFrame.from_dict({'K': [k], 'Trained_accuracy': [Tr_accuracy], 'Tested_accuracy': [Te_accuracy]})
    accuracy_data = pd.concat([accuracy_data, accuracy_values], ignore_index=True)

print(accuracy_data.head())

fig, ax = plt.subplots()
accuracy_data.plot(kind='line', x='K', y='Trained_accuracy', ax=ax, c='red', label='trained')
accuracy_data.plot(kind='line', x='K', y='Tested_accuracy', ax=ax, c='blue', label='tested')
plt.legend(loc='best')
plt.ylabel('Accuracy')
plt.xlabel('species')
plt.show()