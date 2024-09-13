
# Classification with MNIST dataset

#### Author : Hyeri Kim

###Load & check the data

"""
1. Load the MINST data into a pandas dataframe named MINST_firstname where first name is you
name.
2. List the keys
3. Assign the data to a ndarray named X_firstname where firstname is your first name.
4. Assign the target to a variable named y_firstname where firstname is your first name.
5. Print the types of X_firstname and y_firstname.
6. Print the shape of X_firstname and y_firstname.
"""

from sklearn.datasets import fetch_openml

import pandas as pd
import numpy as np

np.random.seed(60)

MNIST_Hyeri = fetch_openml('mnist_784', version = 1)

dir(MNIST_Hyeri)

X_Hyeri = MNIST_Hyeri['data']
y_Hyeri = MNIST_Hyeri['target']

type(X_Hyeri)

type(y_Hyeri)

X_Hyeri.shape

y_Hyeri.shape

"""7. Create three variables named as follows:

  a. If your first name starts by “A” through “L” name the variable some_digit1,
some_digit2, some_digit3. Store in these variables the values from X_firstname indexed
7,5,0 in order.
"""

X_Hyeri = pd.DataFrame(X_Hyeri).to_numpy()

y_Hyeri = pd.DataFrame(y_Hyeri).to_numpy()

some_digit1 = X_Hyeri[7]
some_digit2 = X_Hyeri[5]
some_digit3 = X_Hyeri[0]

"""8. Use imshow method to plot the values of the three variables you defined in the above point.

  Note the values in your Analysis report (written response).
"""

import matplotlib.pyplot as plt

img1 = some_digit1.reshape(28, 28)
img2 = some_digit2.reshape(28, 28)
img3 = some_digit3.reshape(28, 28)
plt.imshow(img1, cmap = "binary")    # cmap ="gray" or ...
plt.axis("off")                      # remove axis
plt.show()
plt.imshow(img2, cmap = "binary")
plt.axis("off")
plt.show()
plt.imshow(img3, cmap = "binary")
plt.axis("off")
plt.show()

"""###Pre-process the data

9. Change the type of y to uint8
10. The current target values range from 0 to 9 i.e. 10 classes. Transform the target variable to 3
classes as follows:

  a. Any digit between 0 and 3 inclusive should be assigned a target value of 0

  b. Any digit between 4 and 6 inclusive should be assigned a target value of 1

  c. Any digit between 7 and 9 inclusive should be assigned a target value of 9

  (Hint: you can use numpy.where to carry out the transformation on the target.)
"""

y_Hyeri = y_Hyeri.astype(np.uint8)
y_Hyeri[:10, :]                     # to check some of original values

y_Hyeri = np.where((y_Hyeri >= 0) & (y_Hyeri <= 3), 0, y_Hyeri)

y_Hyeri = np.where((y_Hyeri >= 4) & (y_Hyeri <= 6), 1, y_Hyeri)

y_Hyeri = np.where((y_Hyeri >= 7) & (y_Hyeri <= 9), 9, y_Hyeri)

y_Hyeri[:10, :]                     # to verify some of changed values

"""11. Print the frequencies of each of the three target classes and note it in your written report in
addition provide a screenshot showing a bar chart.
"""

unique, counts = np.unique(y_Hyeri, return_counts = True)   # to count the target values
print(dict(zip(unique, counts)))                            # to verify frequency

label = ['0 - 3','4 - 6','7 - 9']
plt.bar(unique, counts, tick_label= label)
plt.ylabel('Freuquency')
plt.xlabel('values')
plt.title('Frequency of the classes')
plt.show()

"""12. Split your data into train, test. Assign the first 50,000 records for training and the last 20,000
records for testing. (Hint you don’t need sklearn train test as the data is already randomized).
"""

X_Hyeri_train, y_Hyeri_train = X_Hyeri[:50000], y_Hyeri[:50000]
X_Hyeri_test, y_Hyeri_test = X_Hyeri[50000:], y_Hyeri[50000:]

"""### Build Classification Models
Naive Bayes
13. Train a Naive Bayes classifier using the training data. Name the classifier NB_clf_firstname.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

NB_clf_Hyeri = MultinomialNB()
NB_clf_Hyeri.fit(X_Hyeri_train, y_Hyeri_train)

"""14. Use 3-fold cross validation to validate the training process, and note the results in your written
response.
"""

score = cross_val_score(NB_clf_Hyeri, X_Hyeri_train,  y_Hyeri_train.ravel(), cv = 3, scoring = "accuracy")
score

# mean of Accuracy

mean_accuracy = score.mean()
mean_accuracy

"""15. Use the model to score the accuracy against the test data, note the result in your written
response.
"""

test_score = NB_clf_Hyeri.score(X_Hyeri_test, y_Hyeri_test)
test_score

"""16. Generate the accuracy matrix."""

y_pred = NB_clf_Hyeri.predict(X_Hyeri_test)

cm = confusion_matrix(y_Hyeri_test, y_pred)

cm

"""17. Use the classifier to predict the three variables you defined in point 7 above. Note the results in
your written response and compare against the actual results.
"""

# create function to print the prediction using Naive Bayes Classification

def pred_NB_clf_Hyeri(some_digit):
    pred = NB_clf_Hyeri.predict([some_digit])
    print(pred)

pred_NB_clf_Hyeri(some_digit1)

pred_NB_clf_Hyeri(some_digit2)

pred_NB_clf_Hyeri(some_digit3)

"""Logistic regression

18. Train a Logistic regression classifier using the same training data. Name the classifier
LR_clf_firstname.

  (Note this is a multi-class problem make sure to check all the parameters and
set multi_class='multinomial').

  Try training the classifier using two solvers first “lbfgs” then “Saga”. Set max_iter to 1200 and
tolerance to 0.1 in both cases.
"""

from sklearn.linear_model import LogisticRegression

# Logistic Regression : lbfgs

LR_clf_Hyeri_lbfgs = LogisticRegression(solver = 'lbfgs', multi_class='multinomial', max_iter = 1200, tol = 0.1)
LR_clf_Hyeri_lbfgs.fit(X_Hyeri_train, y_Hyeri_train.ravel())

"""19. Use 3-fold cross validation on the training data and note the results in your written response."""

score_lbfgs = cross_val_score(LR_clf_Hyeri_lbfgs, X_Hyeri_train,  y_Hyeri_train.ravel(), cv = 3, scoring = "accuracy")
score_lbfgs

# mean of Accuracy

mean_accuracy_lbfgs = score_lbfgs.mean()
mean_accuracy_lbfgs

"""20. Use the model to score the accuracy against the test data, note the result in your written
response.
"""

test_score_lbfgs = LR_clf_Hyeri_lbfgs.score(X_Hyeri_test, y_Hyeri_test)
test_score_lbfgs

"""21. Generate the Generate the accuracy matrix precision and recall of the model and note them in
your written response.
"""

y_pred_lbfgs = LR_clf_Hyeri_lbfgs.predict(X_Hyeri_test)

cm_lbfgs = confusion_matrix(y_Hyeri_test, y_pred_lbfgs)
cm_lbfgs

# Logistic Regression : saga

LR_clf_Hyeri_saga = LogisticRegression(solver = 'saga', multi_class='multinomial', max_iter = 1200, tol = 0.1)
LR_clf_Hyeri_saga.fit(X_Hyeri_train, y_Hyeri_train.ravel())

score_saga = cross_val_score(LR_clf_Hyeri_saga, X_Hyeri_train,  y_Hyeri_train.ravel(), cv = 3, scoring = "accuracy")
score_saga

# mean of Accuracy

mean_accuracy_saga = score_saga.mean()
mean_accuracy_saga

test_score_saga = LR_clf_Hyeri_saga.score(X_Hyeri_test, y_Hyeri_test)
test_score_saga

y_pred_saga = LR_clf_Hyeri_saga.predict(X_Hyeri_test)

cm_saga = confusion_matrix(y_Hyeri_test, y_pred_saga)
cm_saga

"""### Classification Report"""

from sklearn.metrics import classification_report

cr_NB = classification_report(y_Hyeri_test, y_pred)
cr_lbfgs = classification_report(y_Hyeri_test, y_pred_lbfgs)
cr_saga = classification_report(y_Hyeri_test, y_pred_saga)

print("1. Naive Bayes Classification\n\n", cr_NB)
print('-' * 60)
print("2. Logistic Regression - lbfgs\n\n", cr_lbfgs)
print('-' * 60)
print("3. Logistic Regression - saga\n\n", cr_saga)

"""22. Use the classifier that worked from the above point to predict the three variables you defined in
point 7 above.

  Note the results in your written response and compare against the actual results
"""

# create function to print the prediction using Logistic Regression - lbfgs Classification

def pred_LR_clf_lbfgs_Hyeri(some_digit):
    pred = LR_clf_Hyeri_lbfgs.predict([some_digit])
    print(pred)

pred_LR_clf_lbfgs_Hyeri(some_digit1)

pred_LR_clf_lbfgs_Hyeri(some_digit2)

pred_LR_clf_lbfgs_Hyeri(some_digit3)

# create function to print the prediction using Logistic Regression - saga Classification

def pred_LR_clf_saga_Hyeri(some_digit):
    pred = LR_clf_Hyeri_saga.predict([some_digit])
    print(pred)

pred_LR_clf_saga_Hyeri(some_digit1)

pred_LR_clf_saga_Hyeri(some_digit2)

pred_LR_clf_saga_Hyeri(some_digit3)

