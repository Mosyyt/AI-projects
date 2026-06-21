import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import numpy as np
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
from sklearn.model_selection import train_test_split
import warnings as w

w.filterwarnings('ignore')

# Load Data
data = pd.read_csv("/AI-Data.csv")

# ---------------------- Graph Menu ---------------------- #
def plot_graph(x_col, title, hue='Class', order=None, hue_order=['L', 'M', 'H']):
    print(f"Loading {title}...\n")
    t.sleep(1)
    plt.figure(figsize=(10, 6))
    sb.countplot(x=x_col, hue=hue, data=data, order=order, hue_order=hue_order)
    plt.title(title)
    plt.show()

graph_options = {
    1: ("Class", "Marks Class Count Graph", None),
    2: ("Semester", "Marks Class Semester-wise Graph"),
    3: ("gender", "Marks Class Gender-wise Graph", ['M', 'F']),
    4: ("NationalITy", "Marks Class Nationality-wise Graph"),
    5: ("GradeID", "Marks Class Grade-wise Graph", 
        ['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12']),
    6: ("SectionID", "Marks Class Section-wise Graph"),
    7: ("Topic", "Marks Class Topic-wise Graph"),
    8: ("StageID", "Marks Class Stage-wise Graph"),
    9: ("StudentAbsenceDays", "Marks Class Absent Days-wise Graph")
}

# Menu
while True:
    print("\n".join([
        "1.Marks Class Count Graph\t2.Marks Class Semester-wise Graph",
        "3.Marks Class Gender-wise Graph\t4.Marks Class Nationality-wise Graph",
        "5.Marks Class Grade-wise Graph\t6.Marks Class Section-wise Graph",
        "7.Marks Class Topic-wise Graph\t8.Marks Class Stage-wise Graph",
        "9.Marks Class Absent Days-wise\t10.No Graph\n"
    ]))
    ch = int(input("Enter Choice: "))
    if ch == 10:
        print("Exiting...\n")
        break
    elif ch in graph_options:
        args = graph_options[ch]
        plot_graph(*args)
    else:
        print("Invalid choice, try again.")

# ---------------------- Data Preprocessing ---------------------- #
cols_to_drop = [
    "gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", "SectionID",
    "Topic", "Semester", "Relation", "ParentschoolSatisfaction",
    "ParentAnsweringSurvey", "AnnouncementsView"
]
data = data.drop(columns=cols_to_drop)

u.shuffle(data)

# Encode categorical columns
for column in data.columns:
    if data[column].dtype == object:
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Features & Labels
feats = data.iloc[:, 0:4].values
lbls = data.iloc[:, 4].values

# Train-Test Split
feats_train, feats_test, lbls_train, lbls_test = train_test_split(feats, lbls, test_size=0.30, shuffle=True)

# ---------------------- Models ---------------------- #
models = {
    "Decision Tree": tr.DecisionTreeClassifier(),
    "Random Forest": es.RandomForestClassifier(),
    "Perceptron": lm.Perceptron(),
    "Logistic Regression": lm.LogisticRegression(),
    "MLP Classifier": nn.MLPClassifier(activation="logistic")
}

# Train & Evaluate
for name, model in models.items():
    model.fit(feats_train, lbls_train)
    preds = model.predict(feats_test)
    acc = m.accuracy_score(lbls_test, preds)
    print(f"\n{name}:")
    print(m.classification_report(lbls_test, preds))
    print(f"Accuracy: {round(acc, 3)}\n")
    t.sleep(1)

# ---------------------- Prediction from User Input ---------------------- #
def class_label(value):
    return {0: "H", 1: "M", 2: "L"}.get(value, "Unknown")

choice = input("Do you want to test specific input (y or n): ")
if choice.lower() == "y":
    rai = int(input("Enter raised hands: "))
    res = int(input("Enter Visited Resources: "))
    dis = int(input("Enter no. of Discussions: "))
    absc = input("Enter No. of Absences (Under-7 or Above-7): ")
    absc = 1 if absc == "Under-7" else 0
    arr = np.array([rai, res, dis, absc]).reshape(1, -1)

    for name, model in models.items():
        pred = class_label(model.predict(arr)[0])
        print(f"Using {name}: {pred}")
        t.sleep(1)
    print("\nExiting...")
else:
    print("Exiting..")
    t.sleep(1)
