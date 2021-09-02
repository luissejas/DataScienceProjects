# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import anderson
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split



###IMPORT DATA
os.chdir("/Users/aayushmarishi/Desktop/Jamie/Intro to Analytics in Business/Group project")
data_ori = pd.read_csv('AvocadoPrices.csv')



###DATA PREPARATION
##Overview of data
print(data_ori.head(5))
print(data_ori.describe())
print(data_ori.dtypes)


##Scan for NaN Values
print(data_ori.isnull().sum())


##Extract the month data from the 'date' variable
data_rpdate = data_ori
data_rpdate['month'] = pd.DatetimeIndex(data_ori['Date']).month
data_rpdate = data_rpdate.drop('Date', axis = 1)


##Convert categorical values
#convert the 'type' variable, which is binary
print(data_rpdate['type'].value_counts())
data_rpdate['type'] = np.where(data_rpdate['type'] == 'conventional', 1, 0)

#convert the 'region' variable using one hot encoding
print(data_rpdate['region'].value_counts())
region_cat = data_ori['region'].value_counts()
region_names = pd.DataFrame(region_cat.index.tolist(), columns = ['Region'])
region_names = region_names.sort_values('Region')
bi_Region = LabelBinarizer()
bi_dummys = bi_Region.fit_transform(data_rpdate['region'])
bi_dummys = pd.DataFrame(bi_dummys, columns = region_names['Region'].tolist())
bi_dummys = bi_dummys.drop('Albany', axis = 1)
data_reg = data_rpdate.drop('region', axis = 1)
data_reg = pd.concat([data_reg, bi_dummys], axis = 1)


##Drop Irrelevant Variables
#the first column of the data is not a variable and thus should be dropped
data_reg = data_reg.drop('Unnamed: 0', axis = 1)



###EXPLORATORY DATA ANALYSIS (EDA)
##Non-graphical EDA
print(data_reg.describe())


##Graphical EDA
#numerical values--pairplot
numericals = pd.concat([data_reg.iloc[:, 0:9], data_reg.iloc[:, 10:12]], axis = 1)
sns.set_style("white")
sns.pairplot(numericals)
plt.show()

#numerical values--boxplot
numericals_nm = (numericals-numericals.min()) / (numericals.max() - numericals.min())
sns.set_style("white")
plt.xticks(rotation = 90)
sns.boxplot(data = numericals_nm, orient = 'v', palette = 'Set2')
plt.show()

#numerical values--correlation matrix
sns.set(rc={'figure.figsize':(9,8)})
corr_matrix = numericals.corr()
sns.heatmap(corr_matrix, cmap="mako", annot = True)
plt.show()

#categorical values--barplot
sns.set_style("white")
sns.countplot(x = "type", data = data_ori)
plt.show()
plt.xticks(rotation = 90)
sns.set(rc={'figure.figsize':(10,7)})
sns.countplot(x = "region", data = data_ori)
plt.show()


##Handling outliers
#deletion of outliers of target variable (top 5%)
Price_95 = np.quantile(data_reg['AveragePrice'],0.95)
outlier_index = data_reg[data_reg['AveragePrice'] > Price_95].index
data_reg.drop(outlier_index, inplace=True)
print(anderson(data_reg['AveragePrice']))

#log transform the highly skewed variables
handle_zeros = data_reg.iloc[:, 1:9]+1
log_transf = pd.DataFrame(handle_zeros.apply(np.log))
sns.set_style("white")
sns.pairplot(log_transf)

data_log = data_reg.drop(data_reg.iloc[:, 1:9], axis = 1)
data_log = pd.concat([data_log, log_transf], axis = 1)


##Drop Highly Correlated Values
#since the "total bags" and "total volume" columns have high linear correlations
    #with other columns, they should be dropped.
data_ = data_log
data_ = data_.drop('Total Bags', axis = 1)
data_ = data_.drop('Total Volume', axis = 1)


##Scaling Data: Normalization
data = data_
data.iloc[:, 2:4] = (data.iloc[:, 2:4] - data.iloc[:, 2:4].min()) / (data.iloc[:, 2:4].max() - data.iloc[:, 2:4].min())
data.iloc[:, 57:63] = (data.iloc[:, 57:63] - data.iloc[:, 57:63].min()) / (data.iloc[:, 57:63].max() - data.iloc[:, 57:63].min())


##Partitioning Data
#separate predictors and target variable
X = data.drop('AveragePrice', axis = 1)
Y = data['AveragePrice']

#separate training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)
print(Y_train.shape)
print(Y_test.shape)


##Others
#create a report for storing results
report = pd.DataFrame(columns=['Model','R2.Train','R2.Test'])

#pre-calculations for R2
Y_train_mean = Y_train.mean()
print("Y_train_mean =", Y_train_mean)
Y_train_meandev = sum((Y_train-Y_train_mean) ** 2)
print("Y_train_meandev =", Y_train_meandev)
Y_test_meandev = sum((Y_test-Y_train_mean) ** 2)
print("Y_test_meandev =", Y_test_meandev)



################
#     OLS      #
################

from sklearn.linear_model import LinearRegression
lmCV = LinearRegression()

from sklearn.model_selection import GridSearchCV
param_grid = {'fit_intercept': [True, False]}
CV_olsmodel = GridSearchCV(estimator=lmCV, param_grid=param_grid, cv=10)
CV_olsmodel.fit(X_train, Y_train)
print(CV_olsmodel.best_params_)

lmCV = lmCV.set_params(**CV_olsmodel.best_params_)
lmCV.fit(X_train, Y_train)

Y_train_pred = lmCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev / Y_train_meandev
print("R2 =", r2)

Y_test_pred = lmCV.predict(X_test)
Y_test_dev = sum((Y_test - Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev / Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['OLS RegressionCV', r2, pseudor2]



####################
# Ridge Regression #
####################

from sklearn.linear_model import Ridge
ridgeregCV = Ridge()

from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.16, 0.2]}
CV_rrmodel = GridSearchCV(estimator = ridgeregCV, param_grid = param_grid, cv = 10)
CV_rrmodel.fit(X_train, Y_train)
print(CV_rrmodel.best_params_)

ridgeregCV = ridgeregCV.set_params(**CV_rrmodel.best_params_)
ridgeregCV.fit(X_train, Y_train)

Y_train_pred = ridgeregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred) ** 2)
r2 = 1 - Y_train_dev / Y_train_meandev
print("R2 =", r2)

Y_test_pred = ridgeregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred) ** 2)
pseudor2 = 1 - Y_test_dev / Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['Ridge RegressionCV', r2, pseudor2]



#############################
# Support Vector Regression #
#############################

from sklearn.svm import SVR
RbfSVRregCV = SVR()

from sklearn.model_selection import GridSearchCV
param_grid = { 
    'kernel': ["rbf"], 
    'C': [12, 14],
    'epsilon': [0.033],
    'gamma' : [1.5]
}
CV_svrmodel = GridSearchCV(estimator=RbfSVRregCV, param_grid=param_grid, cv=10)
CV_svrmodel.fit(X_train, Y_train)
print(CV_svrmodel.best_params_)

RbfSVRregCV = RbfSVRregCV.set_params(**CV_svrmodel.best_params_)
RbfSVRregCV.fit(X_train, Y_train)

Y_train_pred = RbfSVRregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

Y_test_pred = RbfSVRregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['Support Vector RegressionCV', r2, pseudor2]



##################
# Neural Network #
##################

from sklearn.neural_network import MLPRegressor
NNetRregCV = MLPRegressor(solver='lbfgs', max_iter=10000, random_state=0)

from sklearn.model_selection import GridSearchCV
param_grid = { 
    'learning_rate': ["constant", "adaptive"],
    'hidden_layer_sizes': [(25,25,25)],
    'alpha': [0.11],
    'activation': ["tanh"]
}
CV_nnmodel = GridSearchCV(estimator=NNetRregCV, param_grid=param_grid, cv=10)
CV_nnmodel.fit(X_train, Y_train)
print(CV_nnmodel.best_params_)

NNetRregCV = NNetRregCV.set_params(**CV_nnmodel.best_params_)
NNetRregCV.fit(X_train, Y_train)

Y_train_pred = NNetRregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

Y_test_pred = NNetRregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['Neural NetworkCV', r2, pseudor2]



#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestRegressor
RForregCV = RandomForestRegressor(random_state=0)

from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [25., 26.],
    'n_estimators': [310]
}
CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid=param_grid, cv=10)
CV_rfmodel.fit(X_train, Y_train)
print(CV_rfmodel.best_params_)

RForregCV = RForregCV.set_params(**CV_rfmodel.best_params_)
RForregCV.fit(X_train, Y_train)

Y_train_pred = RForregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

Y_test_pred = RForregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['Random ForestCV', r2, pseudor2]



#####################
# Gradient Boosting #
#####################

from sklearn.ensemble import GradientBoostingRegressor
GBoostregCV = GradientBoostingRegressor(random_state=0)

from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [7.],
    'subsample': [0.86],
    'n_estimators': [300],
    'learning_rate': [0.225, 0.25]
}
CV_gbmodel = GridSearchCV(estimator=GBoostregCV, param_grid=param_grid, cv=10)
CV_gbmodel.fit(X_train, Y_train)
print(CV_gbmodel.best_params_)

GBoostregCV = GBoostregCV.set_params(**CV_gbmodel.best_params_)
GBoostregCV.fit(X_train, Y_train)

Y_train_pred = GBoostregCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

Y_test_pred = GBoostregCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['Gradient BoostingCV', r2, pseudor2]



###########################
#  Other Ensemble Methods #
###########################
##The best three models are selected to build an ensemble, in this case, the three
    #models are: Neural Network, Support Vector Regression, and Gradient
    #Boosting.
##The ensemble methods used here are bagging and stacking, and the model with the
    #best result among the previous (Support Vector Regression) is chosen to be
    #the supervisor model
# Bagging #

from sklearn.ensemble import VotingRegressor
reg1 = MLPRegressor(solver='lbfgs', max_iter=10000, random_state=0, alpha=0.11,
                    activation='tanh', hidden_layer_sizes=(25,25,25), 
                    learning_rate='constant')
reg2 = SVR(kernel="rbf", C=14, epsilon=0.033, gamma=1.5)
reg3 = GradientBoostingRegressor(random_state=0, n_estimators=300, max_depth=7,
                                 subsample=0.86, learning_rate=0.225)
VotingReg = VotingRegressor(estimators=[('NeuralNetwork', reg1), 
                                        ('SVR', reg2), ('GradientBoosting', reg3)])
VotingReg.fit(X_train, Y_train)

Y_train_pred = VotingReg.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

Y_test_pred = VotingReg.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['Voting Regressor', r2, pseudor2]


# Stacking #

from mlxtend.regressor import StackingCVRegressor
StackingRegCV = StackingCVRegressor(regressors=(reg1, reg2, reg3), 
                                    meta_regressor=reg1, cv=10)
StackingRegCV.fit(X_train, Y_train)

Y_train_pred = StackingRegCV.predict(X_train)
Y_train_dev = sum((Y_train-Y_train_pred)**2)
r2 = 1 - Y_train_dev/Y_train_meandev
print("R2 =", r2)

Y_test_pred = StackingRegCV.predict(X_test)
Y_test_dev = sum((Y_test-Y_test_pred)**2)
pseudor2 = 1 - Y_test_dev/Y_test_meandev
print("Pseudo-R2 =", pseudor2)

report.loc[len(report)] = ['Stacking RegressorCV', r2, pseudor2]



################
# Final Report #
################

print(report)