import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#_______________________________________________________________________________________________________________________________________

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#target
y_train = train.Survived

#assigning 1 for male and 0 for female
train.Sex = train.Sex.apply(lambda x: 1 if x == 'male' else 0)
test.Sex = test.Sex.apply(lambda x: 1 if x == 'male' else 0)
#______________________________________________________________________________________________________________________________________

#let's find how many missing values we have for each column
#for i in train.columns:
#    print(str(i), " ", str(train[i].isna().sum()))

#for i in test.columns:
#    print(str(i), " ", str(test[i].isna().sum()))

# calculating the mean of age
mean_age_train = train.Age.mean()
mean_age_test = test.Age.mean()

#substituting the missing values for the age with the mean
train.Age = train.Age.apply(lambda x: mean_age_train if pd.isna(x) == True else x)
test.Age = test.Age.apply(lambda x: mean_age_test if pd.isna(x) == True else x)
test.Fare = test.Fare.apply(lambda x: test.Fare.mean() if pd.isna(x) == True else x)

#selecting the features to include
X = np.array([train['Pclass'],train['Sex'],train['Age'],train['SibSp'],train['Parch'],train['Fare']])
X = X.T
y = np.array([train.Survived])
y = y.T


x_final = np.array([test['Pclass'],test['Sex'],test['Age'],test['SibSp'],test['Parch'],test['Fare']])
x_final = x_final.T

#_______________________________________________________________________________________________________________________________________

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.15, random_state = 0)

#let's standardize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
x_final_std = sc.transform(x_final)

#_______________________________________________________________________________________________________________________________________

# Define the parameter grid for each classifier
param_grid_rf = {'max_depth': [3, 5, 7, None],
                 'n_estimators': [50, 100, 200]}

param_grid_lr = {'C': [0.1, 1, 10, 100]}
param_grid_svm = {'C': [0.1, 1, 10, 100]}

#_______________________________________________________________________________________________________________________________________

#let's start with Logistic Regression
lr = LogisticRegression()
# Random Forest Classifier
rf = RandomForestClassifier(random_state=0)
# Support Vector Machine
svm = SVC(probability=True, random_state=0)

# Create a dictionary of classifiers and their corresponding parameter grids
classifiers = {'rf': (rf, param_grid_rf),
               'svm': (svm, param_grid_svm),
               'lr': (lr, param_grid_lr),
               'dt': (DecisionTreeClassifier(random_state=0), {})}

# Perform GridSearchCV for each classifier
for clf_name, (clf, param_grid) in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_std, y_train.ravel())
    print(f"Best parameters for {clf_name}: {grid_search.best_params_}")

best_params_rf = grid_search.best_params_
best_params_svm = grid_search.best_params_
best_params_lr = grid_search.best_params_

best_rf = RandomForestClassifier(**best_params_rf, random_state=0)
best_svm = SVC(**best_params_svm, probability=True, random_state=0)
best_lr = LogisticRegression(**best_params_lr)
best_dt = DecisionTreeClassifier(random_state=0)

#_______________________________________________________________________________________________________________________________________

# Create the final ensemble model
final_voting_clf = VotingClassifier(estimators=[('rf', best_rf), ('svm', best_svm), ('dt', best_dt), ('lr', best_lr)], voting='soft')

# Fit the ensemble model
final_voting_clf.fit(X_train_std, y_train)

# Predictions
y_train_predicted_ensemble = final_voting_clf.predict(X_train_std)
y_test_predicted_ensemble = final_voting_clf.predict(X_test_std)

# Print the results
print('Ensemble Train score: ', accuracy_score(y_train, y_train_predicted_ensemble))
print('Ensemble Test score: ', accuracy_score(y_test, y_test_predicted_ensemble))

#_________________________________________________________________________________________________________________________________________

# Predict the Test set for submission on kaggle

y_final_predicted_ensemble = final_voting_clf.predict(x_final_std)

submission_df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_final_predicted_ensemble})
submission_df.to_csv('submission.csv', index=False,sep = ',')
