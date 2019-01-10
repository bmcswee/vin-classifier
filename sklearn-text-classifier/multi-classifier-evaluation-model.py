# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 10:56:25 2018

@author: Brendan
"""

# Here, several different types of classifiers are tested for their ability to
#   accurately predict model given the VIN as 17 separate features (one for each character of the VIN)

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
from usm_ml_utils import drop_low_samples

# load data
dataFolder = "..\\data-cleaner\\clean-data\\"
df = pd.read_csv(dataFolder + "dataset-fullsplit-make-model.csv")


# count number of examples for each model
modelCount = df['model'].value_counts().sort_index(ascending=False);

# model new dataframe with year count info
df_models = pd.DataFrame(modelCount);
df_models.rename(columns={'model': 'howmany'}, inplace=True); # should refactor so the rename isn't necessary
df_models["whatmodel"] = df_models.index; # convert the indices (in this case, the actual model) to a column
#print(df_years)

# plot the number of examples per year
df_models.plot(x='whatmodel', y='howmany', kind='bar', legend=False, grid=True, figsize=(16,5))
plt.title("Number of samples per car model");
plt.xlabel('model', fontsize=12);
plt.ylabel('Number of samples', fontsize=12);

# Drop all car models with 5 or fewer available training examples
df = drop_low_samples(data=df, label_name='model', drop_below=5)

# split data into test & train sets
X = df.drop(['makemodel', 'vinregex', 'year', 'make', 'model'], axis=1)
y = df['model']
le = LabelEncoder().fit(y) 
labels = le.transform(y)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


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
print("VIN to CAR MODEL")
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
