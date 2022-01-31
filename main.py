import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

digits = load_digits()
dir(digits)

for i in range(10):
    plt.matshow(digits.images[i])

df = pd.DataFrame(digits.data)

df['target'] = digits.target

X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), digits.target,
                                                    test_size=.2)
# create random forest classifier obj, set n_estimators to change n of tree nodes
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train,y_train)
model.score(X_test,y_test)

# create confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test))




