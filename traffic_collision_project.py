# -*- coding: utf-8 -*-
"""Traffic_collision_Group_2_section_003COMP247Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m0JlwI-WD8HBzl3HNxonN1LL0tGbjvLb
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif

# load the dataset
df_Project = pd.read_csv("/content/KSI.csv")

"""## 1. Data exploration"""

print("Types:")
print(df_Project.dtypes)

print("Number of missing values:")
print(df_Project.isnull().sum())
# Print missing value percentage for each feature
print(df_Project.isna().sum()/len(df_Project)*100)

df_Project['ACCLASS'].unique()

print('Description:')
print(df_Project.describe())

print("Range per feature:")
print(df_Project.describe().T[['min','max']])

print("Median:")
print(df_Project.median())

# Compute statistical assessments
print('df_Project.corr()\n',df_Project.corr())

# Visualize the data using histograms
df_Project.hist(bins=20, figsize=(10,10))
plt.show()

# Visualize correlations between features using a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df_Project.corr(), cmap='coolwarm', annot=True)
plt.show()

df_plot = df_Project.DIVISION.value_counts()
df_plot_x = df_plot.index
df_plot_y = df_plot
fig = plt.figure(figsize=(8,6), dpi=120)
ax = fig.add_axes([0,0,1,1])
ax.bar(df_plot_x,df_plot_y)
plt.title(' Reports of crashes on Different Lights')
plt.show()
print(df_plot)

# Use heatmap to find features having maximum missing values
fig, ax = plt.subplots(figsize=(15,7))
sns.heatmap(df_Project.isnull(), yticklabels=False, cmap='Greens')

sns.boxplot(y=df_Project['ACCLASS'], x=df_Project['TIME'])
plt.title('FATAL OR NOT')
plt.show()
print(df_plot)

"""## 2. Data modelling"""

# Replace missing values by NA 
df_Project=df_Project.replace('', np.nan, regex=False)

df_Project.apply(lambda row: row.astype(str).str.contains('NSA').any())

# We are considering the Toronto area, not "Not in Service Area (NSA)" reports, so drop these reports.
df_Project.drop(df_Project[(df_Project['DIVISION'] == 'NSA')].index, inplace=True)
df_Project.drop(df_Project[(df_Project['NEIGHBOURHOOD_140'] == 'NSA')].index, inplace=True)
df_Project['HOOD_140'] = df_Project['HOOD_140'].replace(r'NSA','0').fillna(-1).astype(int)
df_Project.apply(lambda row: row.astype(str).str.contains('NSA').any())

# Drop columns with missing values greater than 50%
df_Project=df_Project.drop(columns=['OFFSET', 'FATAL_NO', 'DRIVACT', 'DRIVCOND',
                                    'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE',
                                    'CYCACT','CYCCOND', 'PEDESTRIAN','CYCLIST',
                                    'MOTORCYCLE','TRUCK', 'TRSN_CITY_VEH',
                                    'EMERG_VEH', 'PASSENGER', 'SPEEDING',
                                    'REDLIGHT', 'ALCOHOL', 'DISABILITY',
                                    'AG_DRIV','INJURY','VISIBILITY','LOCCOORD',
                                    'NEIGHBOURHOOD_158','NEIGHBOURHOOD_140',
                                    'INITDIR', 'MANOEUVER','ACCLOC'])

# Drop columns that are not needed
#Reason for dropping 'LATITUDE','LONGITUDE' are siilar to X, Y so we don't need it.
#automobile only has one value in it.
df_Project = df_Project.drop(columns=['X', 'Y', 'AUTOMOBILE','ObjectId'])

# Fixing YEAR, DATE, TIME columns
df_Project['DATE'] = pd.to_datetime(df_Project['DATE'])
df_Project['MONTH'] = df_Project['DATE'].dt.month
df_Project['DAY'] = df_Project['DATE'].dt.day

# Inserting them back near the front of the columns for easier reference if needed
df_Project.insert(3, 'MONTH', df_Project.pop('MONTH'))
df_Project.insert(4, 'DAY', df_Project.pop('DAY'))

# drop old DATE column
df_Project.drop(columns=['DATE'], inplace=True)

# Display ACCLASS(potential target) column values
df_Project.ACCLASS.unique()

#Combine "non-fatal injury" and "property damage only" to "non-fatal"
df_Project['ACCLASS']=np.where(df_Project['ACCLASS']=='Property Damage Only', 'Non-Fatal', df_Project['ACCLASS'])
df_Project['ACCLASS']=np.where(df_Project['ACCLASS']=='Non-Fatal Injury', 'Non-Fatal', df_Project['ACCLASS'])
df_Project.ACCLASS.unique()
df_Project = df_Project.dropna(subset=['ACCLASS'])
print(df_Project['ACCLASS'].value_counts(normalize=False))



target_remap = {
    'Non-Fatal': 0,
    'Fatal': 1
}
df_Project['ACCLASS'] = df_Project['ACCLASS'].replace(target_remap)

# Change object data type to category
obj_cols=df_Project.select_dtypes(["object"]).columns
df_Project[obj_cols]=df_Project[obj_cols].astype('category')

# Print missing value percentage for each feature
print(df_Project.isna().sum()/len(df_Project)*100)

df_plot = df_Project['ACCLASS'].value_counts(normalize=False) 
labels = ['NON-FATAL', 'FATAL']
colors = sns.color_palette('pastel')[0:2]
plt.pie(df_plot, colors = colors, labels=labels, autopct='%.0f%%') # Matplotlib pie chart
plt.title('FATAL and NON-FATAL')
plt.show()
print(df_plot)

from sklearn.utils import resample

# Declaring classes for forward sampling

df_Project_nonFatal = df_Project[df_Project.ACCLASS == 0]
df_Project_Fatal = df_Project[df_Project.ACCLASS == 1]

# Group the data by ACCLASS and iterate over each group
for name, group in df_Project.groupby('ACCLASS'):
    # Create a histogram for each column in the group
    group.hist(bins=20, figsize=(10,10))
    plt.suptitle(name) # Add a title to each plot showing the ACCLASS
    plt.show()

# Split the dataset into features and target
X = df_Project.drop(columns=['ACCLASS'])
y = df_Project[['ACCLASS']]

# Upsampling "Fatal" class
from sklearn.utils import resample

fatal_upsampled = resample(
    df_Project_Fatal,
    replace=True,
    n_samples=df_Project_nonFatal.shape[0],
    random_state=42
)

# Concatenating 'nonFatal_upsampled' and 'df_Project_Fatal'
df_Project_upsampled = pd.concat([df_Project_nonFatal, fatal_upsampled])

# Verify the result of upsampling
print(df_Project_upsampled.ACCLASS.value_counts())

# Split the dataset into features and target
X = df_Project_upsampled.drop(columns=['ACCLASS'])
y = df_Project_upsampled[['ACCLASS']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

# Split numeric and categorical features
numeric_features = X.select_dtypes(include=["float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["category"]).columns.tolist()

# Define the pre-processing steps for each type of data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine the pre-processing steps into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on the training data
preprocessor.fit(X_train)

# Transform the training and test data
X_train_pp = preprocessor.transform(X_train)
X_test_pp = preprocessor.transform(X_test)

# Fit the selector to the training data
selector = SelectKBest(k=10)
selector.fit(X_train_pp, y_train)
# Get the names of the top 10 selected features
selected_features = selector.get_support()
all_features = numeric_features + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
top_features = [all_features[i] for i in range(len(all_features)) if selected_features[i]]

# Print the names of the top 10 selected features
print("Top 10 selected features:", top_features)
'''
'LATITUDE',
 'STREET1',
 'DISTRICT',
 'DISTRICT',
 'IMPACTYPE',
 'IMPACTYPE',
 'IMPACTYPE',
 'INVTYPE',
 'VEHTYPE',
 'HOOD'
'''

# Select the features with the highest scores
X_train_kbest = selector.transform(X_train_pp)
X_test_kbest = selector.transform(X_test_pp)

"""# Build Predictive Models

"""

# Build the predictive model by random forrest algorithm
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train_kbest, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test_kbest)
accuracy_score(y_test, y_pred)

# Tune model hyperparameters
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [5, 10, 20],
    'n_estimators': [100, 500, 1000]
}

grid_cv = GridSearchCV(clf, params, scoring='accuracy', cv=3)
grid_cv.fit(X_train_kbest, y_train)
print('Best params:', grid_cv.best_params_)
print('Best estimator',grid_cv.best_estimator_)

best_forrest_model=grid_cv.best_estimator_

# Evaluate tuned model
y_pred = best_forrest_model.predict(X_test_kbest)
accuracy_score(y_test, y_pred)

#save this model to pkl file 
import joblib
joblib.dump(best_forrest_model, 'forrest_model_group2.pkl')
print("Model dumped.")

#load the model
best_forrest_model=joblib.load('forrest_model_group2.pkl')

#save training data columns
model_columns=list(X_train_kbest.columns)
joblib.dump(model_columns, 'model_columns_group2.pkl')
print("Model columns dumped.")

#Predictive model building using logistic regression
from sklearn.linear_model import LogisticRegression
log_reg_saga=LogisticRegression(solver="saga", random_state=42)
log_reg_saga.fit(X_train_kbest, y_train)

grid = {'penalty': ['l1', 'l2'], 
        'C':[0.01,0.1, 1,10,100,1000],
        'solver':['saga'],
        'max_iter':[1000]}

grid_search = GridSearchCV(log_reg_saga, grid, scoring ='accuracy',cv=3,return_train_score=True, verbose = 3)
grid_search.fit(X_train_kbest, y_train)

#Best parameters
print('Best parameters',grid_search.best_params_)
print('Best estimator',grid_search.best_estimator_)

best_logistic_model = grid_search.best_estimator_

y_pred_saga = best_logistic_model.predict(X_test_kbest)
accuracy_score(y_test, y_pred_saga)

#Predictive model building using SVM
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

svm=SVC(random_state=42)

param_svm = [
    {'kernel': ['poly','rbf'], 
     'C': [0.01,0.1, 1],
     'gamma': [0.01, 0.05, 0.1],
     'degree':[2,3]}
  ]

random_search_svm= RandomizedSearchCV(svm, param_svm, scoring='accuracy', refit=True, verbose=3)

random_search_svm.fit(X_train_kbest, y_train)

#Best parameters
print(random_search_svm.best_params_)
print(random_search_svm.best_estimator_)

best_rs_svm_model= random_search_svm.best_estimator_

y_pred_rs_svm = best_rs_svm_model.predict(X_test_kbest)
accuracy_score(y_test, y_pred_rs_svm)

# Predictive model building using k neighbors

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)

param_grid_knn = {
    'n_neighbors' : [5, 7, 9, 11, 13, 15],
    'weights' : ['uniform', 'distance']
}

# Perform grid search to find best hyperparameters
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=3)
grid_search_knn.fit(X_train_kbest, y_train)

best_knn_model = grid_search_knn.best_estimator_
y_pred_knn = best_knn_model.predict(X_test_kbest)
accuracy_score(y_test, y_pred_knn)

print(list(X_train_kbest.columns))

import joblib 
joblib.dump(best_knn_model,'bestKNN_Model_group2.pkl')
print("Model Dumped")

model_columns = list(X_train_kbest.columns)
joblib.dump(model_columns,'model_columns_group2.pkl')
print('Models Columns Dumped')





