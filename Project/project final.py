import csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from scipy import stats
from sklearn.feature_selection import RFE

pd.options.mode.chained_assignment = None


### Fairness Metrics

def stat_parity(preds, sens):
    '''
    :preds: numpy array of the model predictions. Consisting of 0s and 1s
    :sens: numpy array of the sensitive features. Consisting of 0s and 1s
    :return: the statistical parity. no need to take the absolute value
    '''

    # TODO. 10pts
    data = pd.DataFrame({'preds': preds, 'sens': sens})

    sens_group = data[data.sens == 1]
    non_sens_group = data[data.sens == 0]

    prob_sens_group = len(sens_group[sens_group.preds == 1]) / len(sens_group)
    prob_non_sens_group = len(non_sens_group[non_sens_group.preds == 1]) / len(non_sens_group)

    result = prob_sens_group - prob_non_sens_group

    return result


def eq_oppo(preds, sens, labels):
    '''
    :preds: numpy array of the model predictions. Consisting of 0s and 1s
    :sens: numpy array of the sensitive features. Consisting of 0s and 1s
    :labels: numpy array of the ground truth labels of the outcome. Consisting of 0s and 1s
    :return: the statistical parity. no need to take the absolute value
    '''

    # TODO. 10pts
    data = pd.DataFrame({'preds': preds, 'sens': sens, 'labels': labels})

    sens_group = data[data.sens == 1]
    non_sens_group = data[data.sens == 0]

    tp = len(sens_group[(sens_group.preds == 1) & (sens_group.labels == 1)])
    fn = len(sens_group[(sens_group.preds == 0) & (sens_group.labels == 1)])
    try:
        tpr_sens_group = tp / (tp + fn)
    except:
        tpr_sens_group = 0

    tp = len(non_sens_group[(non_sens_group.preds == 1) & (non_sens_group.labels == 1)])
    fn = len(non_sens_group[(non_sens_group.preds == 0) & (non_sens_group.labels == 1)])
    try:
        tpr_non_sens_group = tp / (tp + fn)
    except:
        tpr_non_sens_group = 0

    result = tpr_sens_group - tpr_non_sens_group

    return result


## Data Preprocessing

### Preprocessing DataFrame

def process_dfs(df_train_x, df_test_x, categ_cols):
    '''
    Pre-process the features of the training set and the test set, not including the outcome column.
    Convert categorical features (nominal & ordinal features) to one-hot encodings.
    Normalize the numerical features into [0, 1].
    We process training set and the test set together in order to make sure that
    the encodings are consistent between them.
    For example, if one class is encoded as 001 and another class is encoded as 010 in the training set,
    you should follow this mapping for the test set too.

    :df_train: the dataframe of the training data
    :df_test: the dataframe of the test data
    :categ_cols: the column names of the categorical features. the rest features are treated as numerical ones.
    :return: the processed training data and test data, both should be numpy arrays, instead of DataFrames
    '''

    # TODO. 10pts

    df_train_x['source'] = 'train'
    df_test_x['source'] = 'test'
    categ_cols = categ_cols + ['source']

    data = pd.concat([df_train_x, df_test_x])
    data_cat_cols = pd.get_dummies(data[categ_cols], drop_first=True)
    data_num_cols = data.drop(categ_cols, axis=1)
    data_norm_cols = pd.DataFrame(preprocessing.normalize(data_num_cols, axis=0), columns=list(data_num_cols.columns))

    data_processed = pd.concat([data_cat_cols.reset_index(drop=True), data_norm_cols.reset_index(drop=True)], axis=1)
    train_x = data_processed[data_processed.source_train == 1].copy()
    test_x = data_processed[data_processed.source_train == 0].copy()

    train_x.drop('source_train', axis=1, inplace=True)
    test_x.drop('source_train', axis=1, inplace=True)

    return train_x, test_x


por = pd.read_csv('student/student-por.csv', delimiter=';')
por.head()

test = por[['age', 'Medu']]
pd.DataFrame(preprocessing.normalize(test), columns=list(test.columns))

mat = pd.read_csv('student/student-mat.csv', delimiter=';')
mat.head()


data = pd.merge(por, mat,
                on=["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason",
                    "nursery", "internet"], suffixes=['_por', '_mat'])

data.columns






# binary_yn = ['schoolsup_por', 'famsup_por', 'paid_por', 'activities_por', 'nursery', 'higher_por', 'internet',
#        'romantic_por', 'schoolsup_mat', 'famsup_mat', 'paid_mat', 'activities_mat', 'higher_mat', 'romantic_mat']

# for feat in binary_yn:
#     data[feat] = data[feat].map({'yes': 1, 'no': 0})

# protected features
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
data['age'] = data['age'].apply(lambda x: 1 if x > 18 else 0)

data.columns

len(data)

## Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

females = data[data.sex == 0]
males = data[data.sex == 1]
data.columns

plt.hist(data.G3_mat)
plt.xlabel('Final Math Grade')
plt.show()

# calculate a 5-number summary
from numpy import percentile
from numpy.random import rand

# generate data sample
sum_data = data.G3_mat
# calculate quartiles
quartiles = percentile(sum_data, [25, 50, 75])
# calculate min/max
data_min, data_max = sum_data.min(), sum_data.max()
# print 5-number summary
print('Math grades for all students')
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)

np.mean(sum_data)

# generate data sample
sum_data = females.G3_mat
# calculate quartiles
quartiles = percentile(sum_data, [25, 50, 75], interpolation='lower')
# calculate min/max
data_min, data_max = sum_data.min(), sum_data.max()
# print 5-number summary
print('Math grades for female students')
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)

# generate data sample
sum_data = males.G3_mat
# calculate quartiles
quartiles = percentile(sum_data, [25, 50, 75])
# calculate min/max
data_min, data_max = sum_data.min(), sum_data.max()
# print 5-number summary
print('Math grades for male students')
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)

plt.hist(data.G3_por)
plt.xlabel('Final Portuguese Grade')
plt.show()

sum_data = data.G3_por
# calculate quartiles
quartiles = percentile(sum_data, [25, 50, 75])
# calculate min/max
data_min, data_max = sum_data.min(), sum_data.max()
# print 5-number summary
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)

np.mean(sum_data)

sum_data = females.G3_por
# calculate quartiles
quartiles = percentile(sum_data, [25, 50, 75])
# calculate min/max
data_min, data_max = sum_data.min(), sum_data.max()
# print 5-number summary
print('Portuguese grades for female students')
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)

sum_data = males.G3_por
# calculate quartiles
quartiles = percentile(sum_data, [25, 50, 75])
# calculate min/max
data_min, data_max = sum_data.min(), sum_data.max()
# print 5-number summary
print('Portuguese grades for male students')
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)

x = ['Female', 'Male']
y = [len(data[data.sex == 0]), len(data[data.sex == 1])]
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, y)
plt.xlabel('Gender')
plt.ylabel('Number of Students')
plt.xticks(x_pos, x)
plt.title('Student-Gender Distribution')
plt.show()

len(females), len(males)

len(data)

plt.hist(data.G3_por, label='all students')
plt.title('Final Portuguese Grades')
plt.legend(loc='upper left')
plt.show()

females = data[data.sex == 0]
males = data[data.sex == 1]

plt.hist(females.G3_por, label='females')
plt.hist(males.G3_por, label='males')
plt.title('Final Portuguese Grades')
plt.legend(loc='upper left')
plt.savefig('por_hist')
plt.show()

plt.hist(females.G3_mat, label='females')
plt.hist(males.G3_mat, label='males')
plt.title('Final Math Grades')
plt.legend(loc='upper left')
plt.savefig('mat_hist')
plt.show()

data['G3_por_bin'] = data['G3_por'].apply(lambda x: 1 if x > 10 else 0)
data['G3_mat_bin'] = data['G3_mat'].apply(lambda x: 1 if x > 10 else 0)

pd.crosstab(data.G3_por_bin, data.sex, margins=True)

pd.crosstab(data.G3_mat_bin, data.sex, margins=True)

## Preliminary Analysis: Explore fairness in data

# split into training and testing
train, test = train_test_split(data, test_size=0.2)

# mean final grades of protected group SEX for 2 subjects
# portuguese language grades
mean_grade0 = train[train.sex == 0].G3_por.mean()  # females
mean_grade1 = train[train.sex == 1].G3_por.mean()  # males

print('portuguese grade for females: ', mean_grade0)
print('portuguese grade for males: ', mean_grade1)

# math grades
mean_grade0 = train[train.sex == 0].G3_mat.mean()  # females
mean_grade1 = train[train.sex == 1].G3_mat.mean()  # males

print('math grade for females: ', mean_grade0)
print('math grade for males: ', mean_grade1)

# mean final grades of protected group AGE for 2 subjects
# portuguese language grades
mean_grade0 = train[train.age == 0].G3_por.mean()  # older
mean_grade1 = train[train.age == 1].G3_por.mean()  # younger

print('portuguese grade for older students: ', mean_grade0)
print('portuguese grade for younger students: ', mean_grade1)

# math grades
mean_grade0 = train[train.age == 0].G3_mat.mean()  # older
mean_grade1 = train[train.age == 1].G3_mat.mean()  # younger

print('math grade for older students: ', mean_grade0)
print('math grade for younger students: ', mean_grade1)

# t-test p-value for protected groups SEX and AGE for 2 subjects
# portuguese
p_value_sex = stats.ttest_ind(train[train.sex == 0].G3_por, train[train.sex == 1].G3_por)
p_value_age = stats.ttest_ind(train[train.age == 0].G3_por, train[train.age == 1].G3_por)

print('t-test by sex for portuguese', p_value_sex.pvalue)
print('t-test by age for portuguese', p_value_age.pvalue)

# math
p_value_sex = stats.ttest_ind(train[train.sex == 0].G3_mat, train[train.sex == 1].G3_mat)
p_value_age = stats.ttest_ind(train[train.age == 0].G3_mat, train[train.age == 1].G3_mat)

print('t-test by sex for math', p_value_sex.pvalue)
print('t-test by age for math', p_value_age.pvalue)



# t-test p-value for protected groups SEX and AGE for 2 subjects
# portuguese
p_value_sex = stats.ttest_ind(train[train.sex == 0].G3_por_bin, train[train.sex == 1].G3_por_bin)
p_value_age = stats.ttest_ind(train[train.age == 0].G3_por_bin, train[train.age == 1].G3_por_bin)

print('t-test by sex for portuguese', p_value_sex.pvalue)
print('t-test by age for portuguese', p_value_age.pvalue)

# math
p_value_sex = stats.ttest_ind(train[train.sex == 0].G3_mat_bin, train[train.sex == 1].G3_mat_bin)
p_value_age = stats.ttest_ind(train[train.age == 0].G3_mat_bin, train[train.age == 1].G3_mat_bin)

print('t-test by sex for math', p_value_sex.pvalue)
print('t-test by age for math', p_value_age.pvalue)

## Full Analysis: including all features

### Full - Portuguese

# sex
df_train_X = train.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                        axis=1)
df_train_y_por = train.G3_por_bin
df_test_X = test.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                      axis=1)
df_test_y_por = test.G3_por_bin

train_x, test_x = process_dfs(df_train_X, df_test_X,
                              ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                               'guardian_por', 'schoolsup_por', 'famsup_por', 'paid_por', 'activities_por',
                               'nursery', 'higher_por', 'internet', 'romantic_por', 'guardian_mat',
                               'schoolsup_mat', 'famsup_mat', 'paid_mat', 'activities_mat', 'higher_mat',
                               'romantic_mat'])
train_y_por = df_train_y_por.values
test_y_por = df_test_y_por.values


model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('PORTUGUESE baseline: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

# Feature Selection

columns

## RFE - Portuguese

data_final_vars = columns.values.tolist()
y = ['G3_por_bin']
X = [i for i in data_final_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=10)
rfe = rfe.fit(train_x, train_y_por.ravel())
print(rfe.support_)
print(rfe.ranking_)

columns[rfe.support_]

cols = columns[rfe.support_]

# cols=['address_U', 'famsize_LE3', 'Mjob_health', 'Mjob_services', 'Mjob_teacher', 'Fjob_teacher',
#       'reason_home', 'reason_other', 'reason_reputation', 'higher_por_yes',
#       'paid_mat_yes', 'higher_mat_yes', 'Dalc_por', 'absences_por',
#       'Dalc_mat', 'absences_mat']
X = os_data_X[cols]
y = os_data_y['y']

import statsmodels.api as sm

logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())

# sex
df_train_X = train[
    ['sex', 'school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat', 'failures_mat', 'failures_por',
     'absences_por']]
df_train_y_por = train.G3_por_bin
df_test_X = test[
    ['sex', 'school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat', 'failures_mat', 'failures_por',
     'absences_por']]
df_test_y_por = test.G3_por_bin

train_x, test_x = process_dfs(df_train_X, df_test_X,
                              ['sex', 'school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat'])

train_y_por = df_train_y_por.values
test_y_por = df_test_y_por.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('PORTUGUESE RFE: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### Full - Math

# load the data
# sex
df_train_X = train.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                        axis=1)
df_train_y_mat = train.G3_mat_bin
df_test_X = test.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                      axis=1)
df_test_y_mat = test.G3_mat_bin

train_x, test_x = process_dfs(df_train_X, df_test_X,
                              ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                               'guardian_por', 'schoolsup_por', 'famsup_por', 'paid_por', 'activities_por',
                               'nursery', 'higher_por', 'internet', 'romantic_por', 'guardian_mat',
                               'schoolsup_mat', 'famsup_mat', 'paid_mat', 'activities_mat', 'higher_mat',
                               'romantic_mat'])
train_y_mat = df_train_y_mat.values
test_y_mat = df_test_y_mat.values

# Initiate and run the model
model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_mat)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('MATH baseline: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### RFE - Math

data_final_vars = columns.values.tolist()
y = ['G3_mat_bin']
X = [i for i in data_final_vars if i not in y]

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=11)
rfe = rfe.fit(train_x, train_y_mat.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols = columns[rfe.support_]
cols

cols = cols.drop(['guardian_por_other'])
cols

# cols=['sex', 'school_MS', 'address_U', 'Pstatus_T', 'Mjob_health', 'Mjob_other', 'Mjob_services',
#       'Fjob_teacher', 'reason_reputation', 'guardian_por_other', 'higher_por_yes', 'guardian_mat_other',
#       'schoolsup_mat_yes', 'higher_mat_yes', 'Fedu', 'goout_por', 'Walc_por', 'absences_por',
#       'failures_mat', 'absences_mat']
X = os_data_X[cols]
y = os_data_y['y']

import statsmodels.api as sm

logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary2())

# sex
df_train_X = train[
    ['school', 'sex', 'Mjob', 'higher_por', 'absences_mat', 'failures_mat', 'schoolsup_mat', 'higher_mat',
     'failures_por']]
df_train_y_mat = train.G3_mat_bin
df_test_X = test[
    ['school', 'sex', 'Mjob', 'higher_por', 'absences_mat', 'failures_mat', 'schoolsup_mat', 'higher_mat',
     'failures_por']]
df_test_y_mat = test.G3_mat_bin

train_x, test_x = process_dfs(df_train_X, df_test_X,
                              ['school', 'sex', 'Mjob', 'higher_por', 'schoolsup_mat', 'higher_mat'])

train_y_mat = df_train_y_mat.values
test_y_mat = df_test_y_mat.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_mat)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_mat = df_test_sens_feat.values
acc = accuracy_score(test_y_mat, preds)
stat_p = stat_parity(preds, test_sens_mat)
eq_op = eq_oppo(preds, test_sens_por, test_y_mat)
print('MATH RFE: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

## Exploring ways to mitigate bias

### Remove the protected attribute

### Portuguese

# sex
df_train_X = train.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                        axis=1)
df_train_y_por = train.G3_por_bin
df_test_X = test.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                      axis=1)
df_test_y_por = test.G3_por_bin

df_train_X_no_gender = df_train_X.drop(['sex'], axis=1)
df_test_X_no_gender = df_test_X.drop(['sex'], axis=1)

train_x, test_x = process_dfs(df_train_X_no_gender, df_test_X_no_gender,
                              ['school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                               'guardian_por', 'schoolsup_por', 'famsup_por', 'paid_por', 'activities_por',
                               'nursery', 'higher_por', 'internet', 'romantic_por', 'guardian_mat',
                               'schoolsup_mat', 'famsup_mat', 'paid_mat', 'activities_mat', 'higher_mat',
                               'romantic_mat'])

train_y_por = df_train_y_por.values
test_y_por = df_test_y_por.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('PORTUGUESE FULL sex dropped: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### with RFE variables

# sex
df_train_X = train[
    ['sex', 'school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat', 'failures_mat', 'failures_por',
     'absences_por']]
df_train_y_por = train.G3_por_bin
df_test_X = test[
    ['sex', 'school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat', 'failures_mat', 'failures_por',
     'absences_por']]
df_test_y_por = test.G3_por_bin

df_train_X_no_gender = df_train_X.drop(['sex'], axis=1)
df_test_X_no_gender = df_test_X.drop(['sex'], axis=1)

train_x, test_x = process_dfs(df_train_X_no_gender, df_test_X_no_gender,
                              ['school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat'])
train_y_por = df_train_y_por.values
test_y_por = df_test_y_por.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('PORTUGUESE RFE sex dropped: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### Math

# sex
df_train_X = train.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                        axis=1)
df_train_y_mat = train.G3_mat_bin
df_test_X = test.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                      axis=1)
df_test_y_mat = test.G3_mat_bin

df_train_X_no_gender = df_train_X.drop(['sex'], axis=1)
df_test_X_no_gender = df_test_X.drop(['sex'], axis=1)

train_x, test_x = process_dfs(df_train_X_no_gender, df_test_X_no_gender,
                              ['school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                               'guardian_por', 'schoolsup_por', 'famsup_por', 'paid_por', 'activities_por',
                               'nursery', 'higher_por', 'internet', 'romantic_por', 'guardian_mat',
                               'schoolsup_mat', 'famsup_mat', 'paid_mat', 'activities_mat', 'higher_mat',
                               'romantic_mat'])

train_y_mat = df_train_y_mat.values
test_y_mat = df_test_y_mat.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_mat)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_mat = df_test_sens_feat.values
acc = accuracy_score(test_y_mat, preds)
stat_p = stat_parity(preds, test_sens_mat)
eq_op = eq_oppo(preds, test_sens_por, test_y_mat)
print('MATH FULL sex dropped: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### With RFE Variables

# sex
df_train_X = train[
    ['school', 'sex', 'Mjob', 'higher_por', 'absences_mat', 'failures_mat', 'schoolsup_mat', 'higher_mat',
     'failures_por']]
df_train_y_mat = train.G3_mat_bin
df_test_X = test[
    ['school', 'sex', 'Mjob', 'higher_por', 'absences_mat', 'failures_mat', 'schoolsup_mat', 'higher_mat',
     'failures_por']]
df_test_y_mat = test.G3_mat_bin

df_train_X_no_gender = df_train_X.drop(['sex'], axis=1)
df_test_X_no_gender = df_test_X.drop(['sex'], axis=1)

train_x, test_x = process_dfs(df_train_X, df_test_X,
                              ['school', 'Mjob', 'higher_por', 'schoolsup_mat',
                               'higher_mat'])  ### with RFE variables
train_y_mat = df_train_y_mat.values
test_y_mat = df_test_y_mat.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_mat)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_mat = df_test_sens_feat.values
acc = accuracy_score(test_y_mat, preds)
stat_p = stat_parity(preds, test_sens_mat)
eq_op = eq_oppo(preds, test_sens_por, test_y_mat)
print('MATH RFE sex dropped: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

## Augment the training set

### Portuguese

df_train_X = train.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                        axis=1)
df_train_y_por = train.G3_por_bin
df_test_X = test.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                      axis=1)
df_test_y_por = test.G3_por_bin

df_train_X_syn = df_train_X.copy()
df_train_X_syn['sex'] = 1 - df_train_X_syn.sex
df_train_y_syn_por = df_train_y_por.copy()

df_train_X_aug = pd.concat((df_train_X, df_train_X_syn))
df_train_y_aug_por = pd.concat((df_train_y_por, df_train_y_syn_por))

train_x, test_x = process_dfs(df_train_X_aug, df_test_X,
                              ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                               'guardian_por', 'schoolsup_por', 'famsup_por', 'paid_por', 'activities_por',
                               'nursery', 'higher_por', 'internet', 'romantic_por', 'guardian_mat',
                               'schoolsup_mat', 'famsup_mat', 'paid_mat', 'activities_mat', 'higher_mat',
                               'romantic_mat'])

train_y_por = df_train_y_aug_por.values
test_y_por = df_test_y_por.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('PORTUGUESE FULL AUG: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### with RFE variables

# sex
df_train_X = train[
    ['sex', 'school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat', 'failures_mat', 'failures_por',
     'absences_por']]
df_train_y_por = train.G3_por_bin
df_test_X = test[
    ['sex', 'school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat', 'failures_mat', 'failures_por',
     'absences_por']]
df_test_y_por = test.G3_por_bin

df_train_X_syn = df_train_X.copy()
df_train_X_syn['sex'] = 1 - df_train_X_syn.sex
df_train_y_syn_por = df_train_y_por.copy()

df_train_X_aug = pd.concat((df_train_X, df_train_X_syn))
df_train_y_aug_por = pd.concat((df_train_y_por, df_train_y_syn_por))

train_x, test_x = process_dfs(df_train_X_aug, df_test_X,
                              ['school', 'Mjob', 'Fjob', 'higher_por', 'higher_mat', 'schoolsup_mat'])

train_y_por = df_train_y_aug_por.values
test_y_por = df_test_y_por.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('PORTUGUESE RFE AUG: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### Math

df_train_X = train.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                        axis=1)
df_train_y_mat = train.G3_mat_bin
df_test_X = test.drop(['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por', 'G3_mat_bin', 'G3_por_bin'],
                      axis=1)
df_test_y_mat = test.G3_mat_bin

df_train_X_syn = df_train_X.copy()
df_train_X_syn['sex'] = 1 - df_train_X_syn.sex
df_train_y_syn_mat = df_train_y_mat.copy()

df_train_X_aug = pd.concat((df_train_X, df_train_X_syn))
df_train_y_aug_mat = pd.concat((df_train_y_mat, df_train_y_syn_mat))

train_x, test_x = process_dfs(df_train_X_aug, df_test_X,
                              ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                               'guardian_por', 'schoolsup_por', 'famsup_por', 'paid_por', 'activities_por',
                               'nursery', 'higher_por', 'internet', 'romantic_por', 'guardian_mat',
                               'schoolsup_mat', 'famsup_mat', 'paid_mat', 'activities_mat', 'higher_mat',
                               'romantic_mat'])

train_y_por = df_train_y_aug_mat.values
test_y_por = df_test_y_mat.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('MATH FULL AUG: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')

### with RFE variables

# sex
df_train_X = train[
    ['school', 'sex', 'Mjob', 'higher_por', 'absences_mat', 'failures_mat', 'schoolsup_mat', 'higher_mat',
     'failures_por']]
df_train_y_mat = train.G3_mat_bin
df_test_X = test[
    ['school', 'sex', 'Mjob', 'higher_por', 'absences_mat', 'failures_mat', 'schoolsup_mat', 'higher_mat',
     'failures_por']]
df_test_y_mat = test.G3_mat_bin

df_train_X_syn = df_train_X.copy()
df_train_X_syn['sex'] = 1 - df_train_X_syn.sex
df_train_y_syn_mat = df_train_y_mat.copy()

df_train_X_aug = pd.concat((df_train_X, df_train_X_syn))
df_train_y_aug_mat = pd.concat((df_train_y_mat, df_train_y_syn_mat))

train_x, test_x = process_dfs(df_train_X_aug, df_test_X,
                              ['school', 'Mjob', 'higher_por', 'schoolsup_mat',
                               'higher_mat'])  ### with RFE variables

train_y_por = df_train_y_aug_mat.values
test_y_por = df_test_y_mat.values

model = LogisticRegression(random_state=0, max_iter=1000)
fit_model = model.fit(train_x, train_y_por)
preds = fit_model.predict(test_x)

df_test_sens_feat = test['sex']
test_sens_por = df_test_sens_feat.values
acc = accuracy_score(test_y_por, preds)
stat_p = stat_parity(preds, test_sens_por)
eq_op = eq_oppo(preds, test_sens_por, test_y_por)
print('MATH RFE AUG: accuracy, statistical parity, and equal opportunity wrt sex')
print(acc, stat_p, eq_op)
print('\n')