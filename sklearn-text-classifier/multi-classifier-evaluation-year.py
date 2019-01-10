
# coding: utf-8

# In[20]:

# Here, several different types of classifiers are tested for their ability to
#   accurately predict model year given the VIN as 17 separate features (one for each character of the VIN)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


# load data
dataFolder = "..\\data-cleaner\\clean-data\\"
df = pd.read_csv(dataFolder + "dataset-fullsplit-make-model.csv")

# grab column names. just practice
#columnnames = list(df.columns.values);
#print(columnnames)

# count number of examples for each model year
modelYearCount = df['year'].value_counts().sort_index(ascending=False);
#print(modelYearCount);

# make new dataframe with year count info
df_years = pd.DataFrame(modelYearCount);
df_years.rename(columns={'year': 'howmany'}, inplace=True); # should refactor so the rename isn't necessary
df_years["whatyear"] = df_years.index; # convert the indices (in this case, the actual year) to a column
#print(df_years)

# plot the number of examples per year
df_years.plot(x='whatyear', y='howmany', kind='bar', legend=False, grid=True, figsize=(8,5))
plt.title("Number of samples per year");
plt.xlabel('Year', fontsize=12);
plt.ylabel('Number of samples', fontsize=12);

# split data into test & train sets
X = df.drop(['makemodel', 'vinregex', 'year', 'make', 'model'], axis=1)
y = df['year']
le = LabelEncoder().fit(y) 
labels = le.transform(y)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Debugging
#print(X_train)    
#print(y_train)
#print(X.head(1))
#print(y.head(1))


# test a bunch of different classifiers
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis()]

# fit and then print results
print("VIN to YEAR")
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
# end for loop 
    

# that's all
