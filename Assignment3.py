import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score, precision_score, confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('fraud_data.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 

def answer_one():
    ans = df['Class'].mean()
    
    return ans


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

def answer_two():
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    
    acc_score = dummy_majority.score(X_test, y_test)
    rec_score = recall_score(y_test, dummy_majority.predict(X_test))
    
    return acc_score, rec_score


# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

def answer_three():
    svc = SVC().fit(X_train, y_train)
    
    acc_score = svc.score(X_test, y_test)
    rec_score = recall_score(y_test, svc.predict(X_test))
    prec_score = precision_score(y_test, svc.predict(X_test))
    
    return acc_score, rec_score, prec_score


# ### Question 4
# 
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

def answer_four():
    parameters = {'C': 1e9, 'gamma': 1e-07}
    svc = SVC(C=parameters['C'], gamma=parameters['gamma']).fit(X_train, y_train)
    predicted = svc.decision_function(X_test) > -220
    
    ans = confusion_matrix(y_test, predicted)
    
    return ans


answer_four()


# ### Question 5
# 
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# 
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

def answer_five():
    lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

    plot_precision_recall_curve(lr, X_test, y_test)
    plt.grid()
    plot_roc_curve(lr, X_test, y_test)
    plt.grid()

    plt.show()
    
    return 0.82, 0.99  # By analysing the graph


# ### Question 6
# 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.
# 
# `'penalty': ['l1', 'l2']`
# 
# `'C':[0.01, 0.1, 1, 10, 100]`
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
# 
# <br>
# 
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*

def answer_six():
    param_grid = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
    lr = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid=param_grid, scoring='recall').fit(X_train, y_train)

    res_test = [{"mean_test_score": result, "C": test['C'], "penalty": test['penalty']} for result, test in
                zip(lr.cv_results_['mean_test_score'], lr.cv_results_['params'])]
    da = pd.DataFrame()
    for element in res_test:
        da.loc[element['C'], element['penalty']] = element['mean_test_score']

    ans = np.array(da)

    return ans


if __name__ == '__main__':
    print('Ex1:\n', answer_one(), end='\n\n\n')
    print('Ex2:\n', answer_two(), end='\n\n\n')
    print('Ex3:\n', answer_three(), end='\n\n\n')
    print('Ex4:\n', answer_four(), end='\n\n\n')
    print('Ex5:\n', answer_five(), end='\n\n\n')
    print('Ex6:\n', answer_six(), end='\n\n\n')


