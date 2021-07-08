import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def encode(df, col):
    encoder = LabelEncoder()
    encoder.fit(df[col])
    new_column = encoder.transform(df[col])

    test = np.unique(new_column)
    result = encoder.inverse_transform(test)
    for i in range(len(test)):
        print(result[i], ":", test[i])

    return new_column


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# combining train and test datasets, will divide into train/test later
df = train.append(test)

# dropping non-important columns (IDs)
df = df.drop(columns=[df.columns[0], "id"], axis=1)
# dropping "Departure Delay in Minutes" column as correlation matrix showed strong correlation (0.96) between them
df = df.drop('Departure Delay in Minutes', axis=1)

# replacing NaN values with 0 -> there are null values only in column "Arrival Delay in Minutes"
df.fillna(0, inplace=True)

# the result of using LabelEncoder
print("-----------LabelEncoder-----------")
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encode(df, col)
    else:
        continue

# making sure every column was of the same type (int32 or int64)
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].astype(int)

# Final form of data
print("Shape: ", df.shape)
print (df.info())
print(df.isnull().sum())


satisfaction_columns = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",]
# for col in satisfaction_columns:
#     var = df[col]
#     var_value = var.value_counts()
#
#     plt.figure(figsize=(9, 4))
#     plt.bar(var_value.index, var_value.values)
#     plt.xlabel("Satisfaction level")
#     plt.ylabel("Frequency")
#     plt.title(col)
#     plt.show()

# satisfaction plot
satisfaction_plot = sns.countplot(x='satisfaction', data=df, order=df["satisfaction"].value_counts().index)
# satisfaction plot vs gender
sns.catplot(x="satisfaction", col="Gender", col_wrap=2, data=df, kind="count", height=3.5, aspect=1.0, palette="Set1")
# satisfaction plot vs passenger type
sns.catplot(x="satisfaction", col="Customer Type", col_wrap=2, data=df, kind="count", height=3.5, aspect=1.0, palette="Set2")
# satisfaction plot vs flight class
sns.catplot(x="satisfaction", col="Class", col_wrap=3, data=df, kind="count", height=3.5, aspect=1.0, palette="Set3")
# satisfaction plot vs type of flight
sns.catplot(x="satisfaction", col="Type of Travel", col_wrap=2, data=df, kind="count", height=3.5, aspect=1.0, palette="Set1");

# correlation matrix
corr_matrix = df.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

# dissatisfaction line plot
plt.figure(figsize=(20, 10))
for feature in satisfaction_columns:
    fig = sns.lineplot(data=df.loc[df['satisfaction'] == 0, feature].value_counts(sort=False), linewidth=2, label=feature)
fig.figure.suptitle("Neutral or dissatisfied vs questionnaire features", fontsize=16)

# satisfaction line plot
plt.figure(figsize=(20, 10))
for feature in satisfaction_columns:
    fig = sns.lineplot(data=df.loc[df['satisfaction'] == 1, feature].value_counts(sort=False), linewidth=2, label=feature)
fig.figure.suptitle("satisfied vs questionnaire features", fontsize=16)

plt.show()


X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


#Logistic Regression
print("---------- LOGISTIC REGRESSION ----------")
log_r = LogisticRegression(max_iter=10000)
log_r.fit(X_train, y_train)
logr_scores = cross_val_score(log_r, X, y, scoring='accuracy', n_jobs=-1)
print("Logistic Regression Mean Accuracy (post cross-validation): ", logr_scores.mean())

plot_confusion_matrix(log_r, X, y)
plt.title("Logistic Regression Confusion Matrix")
plt.show()



#Random Forest
print("---------- Random Forest ----------")
random_forest = RandomForestClassifier(max_depth=25, min_samples_leaf=1, min_samples_split=2,n_estimators=1200, random_state=42)
random_forest.fit(X_train, y_train)
randforest_scores = cross_val_score(random_forest, X, y, scoring='accuracy', n_jobs=-1)
print("Random Forest Mean Accuracy (post cross-validation): ", randforest_scores.mean())

plot_confusion_matrix(random_forest, X, y)
plt.title("Random Forest Confusion Matrix")
plt.show()


#MLP Classifier
print("---------- MLP Classifier ----------")
mlp = MLPClassifier(random_state=42, max_iter=10000)
mlp.fit(X_train, y_train)
mlp_scores = cross_val_score(mlp, X, y, scoring='accuracy', n_jobs=-1)
print("MLP Classifier Mean Accuracy (post cross-validation): ", mlp_scores.mean())

plot_confusion_matrix(mlp, X, y)
plt.title("MLP Classifier Confusion Matrix")
plt.show()

# mean scores plot
models = [str(log_r), str(random_forest), str(mlp)]
mean_scores = [logr_scores.mean(), randforest_scores.mean(), mlp_scores.mean()]
plt.bar(models, mean_scores, label="Cross-Validation Score")
plt.legend(loc="upper left")
plt.xlabel("Model")
plt.ylabel("Cross-Validation Score")
plt.title("Cross-Validation Score of Models")
plt.show()