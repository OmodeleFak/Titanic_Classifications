# %% read data
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survived distribution
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Survived", hue = "Sex", data=train)
plt.title ("Distribution of Survivors by Sex")



# %% Survived w.r.t Pclass / Sex / Embarked ?

sns.countplot(x="Survived", hue = "Pclass", data=train)
plt.title ("Distribution of Survivors by Pclass")

#%% Survived w.r.t Embarked?
sns.countplot(x="Survived", hue = "Embarked", data=train)
plt.title ("Distribution of Survivors by Embarked")

# %% Age distribution ?
sns.countplot(x="Age", data=train)
plt.title ("Distribution of survivors by age group")

def to_age_group(Age):
    if Age > 0 and Age <= 12:
        return "Children"
    elif Age > 12 and Age <= 17:
        return "Adolescents"
    elif Age > 17 and Age <= 64:
        return "Adults"
    else:
        return "Older Adults"


train["Age group"] = train["Age"].apply(to_age_group)

sns.countplot(
    data=train,
    x="Age group",
    order=[
        "Children",
        "Adolescents",
        "Adults",
        "Older Adults",
    ],
)

# %% Survived w.r.t Age distribution ?
sns.countplot(x="Age", data=train)
plt.title ("Distribution of Survivors by Age Range")

def to_age_group(Age):
    if Age > 0 and Age <= 10:
        return "1-10"
    elif Age > 10 and Age <= 20:
        return "11-20"
    elif Age > 20 and Age <= 30:
        return "21-30"
    elif Age > 30 and Age <= 40:
        return "31-40"
    elif Age > 40 and Age <= 50:
        return "41-50"
    elif Age > 50 and Age <= 60:
        return "51-60"
    elif Age > 60 and Age <= 70:
        return "61-70"
    elif Age > 70 and Age <= 80:
        return "71-80"
    elif Age > 80 and Age <= 90:
        return "81-90"
    else:
        return "91-100"


train["Age Range"] = train["Age"].apply(to_age_group)

sns.countplot(
    data=train,
    x="Age Range",
    order=[
        "1-10",
        "11-20",
        "21-30",
        "31-40",
        "41-50",
        "51-60",
        "61-70",
        "71-80",
        "81-90",
        "91-100",
    ],
)


# %% Survived w.r.t SibSp / Parch  ?

sns.barplot(x="Survived", y = "SibSp", hue = "Parch", data=train)
plt.title ("Distribution of survivors by Relationship")

# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %%
# %% Using Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define the list of features you want to use for your model
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']

from sklearn.impute import SimpleImputer

# Create an imputer for numerical features
numerical_imputer = SimpleImputer(strategy='median')
X_train = numerical_imputer.fit_transform(X_train)
X_val = numerical_imputer.transform(X_val)

# Model Selection and Training
clf = LogisticRegression(random_state=2020)
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print(classification_report(y_val, y_pred))


# %%Using Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
clf = RandomForestClassifier(random_state=2020)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the validation data
y_pred = clf.predict(X_val)

# Evaluate the classifier
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print(classification_report(y_val, y_pred))

# %%Using Support Vector Mechanism

from sklearn.svm import SVC

# Create an SVM classifier
clf = SVC(random_state=2020)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the validation data
y_pred = clf.predict(X_val)

# Evaluate the classifier
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print(classification_report(y_val, y_pred))


# %%Using Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier

# Create a Gradient Boosting classifier
clf = GradientBoostingClassifier(random_state=2020)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the validation data
y_pred = clf.predict(X_val)

# Evaluate the classifier
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print(classification_report(y_val, y_pred))

# %%Using KNN

from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier with a specified number of neighbors (k)
k = 5  # You can adjust the value of k
clf = KNeighborsClassifier(n_neighbors=k)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the validation data
y_pred = clf.predict(X_val)

# Evaluate the classifier
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print(classification_report(y_val, y_pred))



# %%
