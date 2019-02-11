'''CREDIT SCORE MODELLING'''

# importing required libraries
from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# reading dataframe
DF = pd.read_csv('C:/Users/PALLAVI/Desktop/BEUTH_1ST_SEM/pet_project/Banking.csv')
print(DF.shape)
print(list(DF.columns))
print(DF.isnull().sum().sum())

# Target Variable
DF['Bad_label'].value_counts()
sns.countplot(x='Bad_label', data=DF, palette='hls')
plt.savefig('count_plot')
plt.show()

# Dummy Varibales-onehot conversion
CHAR_VAR = ['job', 'marital', 'education', 'default',
            'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for var in CHAR_VAR:
    cat_list = pd.get_dummies(DF[var], prefix=var)
    DF_updated = DF.join(cat_list)
    DF = DF_updated

CHAR_VAR = ['job', 'marital', 'education', 'default',
            'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
DF_VARS = DF.columns.values.tolist()
TO_KEEP = [i for i in DF_VARS if i not in CHAR_VAR]
DATA_FINAL = DF[TO_KEEP]

# dependent (X) and independent (Y) variables
X = DATA_FINAL.drop('Bad_label', axis=1)
Y = DATA_FINAL.Bad_label
FEATURE_LABELS = DATA_FINAL.columns[1:]

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Train A Random Forest Classifier

#clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
#clf.fit(X_train, Y_train)
#for feature in zip(FEATURE_LABELS, clf.feature_importances_):
    #print(feature)
# most 10 important features
# 'campaign', 0.27728974589153127 - 2
# 'nr_employed', 0.09050836563990655 -9
# 'duration', 0.07945563510149121 - 1
# 'Bad_label', 0.05016257670167846 - 10
# 'pdays', 0.03841490639937689 - 3
# 'previous', 0.029742359761027973 -4
# 'euribor3m', 0.027031772743079746 -8
# 'cons_price_idx', 0.023344748847786083 -6
# 'cons_conf_idx', 0.022776755597658845 -7
# 'poutcome_success', 0.020471302462824574 -63
# 'housing_yes', 0.01350307854627908 -40
# 'housing_no', 0.013470781739118977 - 38

# drop the features that we do not need.
DATA_FINAL.drop(DATA_FINAL.columns[[0, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                    35, 36, 37, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]],
                axis=1, inplace=True)
# Check the independence between the independent variables
sns.heatmap(DATA_FINAL.corr())
plt.show()

# dependent and independent variables
X_final = DATA_FINAL.drop('Bad_label', axis=1)
Y_final = DATA_FINAL.Bad_label
X_final_train, X_final_test, Y_final_train, Y_final_test = \
    train_test_split(X_final, Y_final, test_size=0.3)

# Logistic Regression Model
CLASSIFIER = LogisticRegression(random_state=0)
CLASSIFIER.fit(X_final_train, Y_final_train)
Y_PREDICT = CLASSIFIER.predict(X_final_test)

# accuracy, precision, recall, F-measure and support
CONFUSION_MATRIX = confusion_matrix(Y_final_test, Y_PREDICT)
print(CONFUSION_MATRIX)
print('Accuracy of LR classifier: {:.2f}'.format(CLASSIFIER.score(X_final_test, Y_final_test)))
print(classification_report(Y_final_test, Y_PREDICT))

def testing(out):
    out.write("Worked fine\n")