# %%
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Read dataset
df = pd.read_csv("1_lebron_james_shot_chart_1_2023.csv")
print(f"Initial columns: {df.columns.tolist()}\n")

# Drop columns
columns_to_drop = [
    "season",
    "color",
    "opponent_team_score",
    "date",
    "qtr",
    "time_remaining",
    "lebron_team_score",
    "lead",
    "opponent",
    "team",
]

df.drop(columns_to_drop, axis=1, inplace=True)
print(f"Final head:\n{df.head()}")

# %%
# Split data
X = df[["top", "left", "shot_type", "distance_ft"]]
y = df.result

print(X.head())
print("\n", y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=16
)


# %%
# Initialize model
logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# %%
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: {cnf_matrix}")

class_names = ["Miss", "Make"]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

# %%
# Show metrics
target_names = ['Shot Miss', 'Shot Make']
print(classification_report(y_test, y_pred, target_names=target_names))


