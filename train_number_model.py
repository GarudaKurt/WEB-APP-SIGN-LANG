import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = os.path.join("data", "numbers", "fsl_number_data.csv")
MODEL_FILE = os.path.join("models", "numbers", "fsl_number_model.joblib")

if not os.path.exists(DATA_FILE):
    print("No number dataset found.")
    print("Run collect_number_data.py first.")
    exit()

try:
    df = pd.read_csv(DATA_FILE, encoding="utf-8", encoding_errors="replace")
except Exception as e:
    print("Error reading number dataset:")
    print(e)
    exit()

if df.empty:
    print("Number dataset is empty. Collect number samples first.")
    exit()

df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"].isin(list("0123456789"))]

if df.empty:
    print("No valid number labels found.")
    print("Valid labels are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.")
    exit()

print("Samples per number:")
print(df["label"].value_counts().sort_index())

X = df.drop(columns=["label"])
y = df["label"]

if y.nunique() < 2:
    print("")
    print("You only collected one number.")
    print("Collect at least 2 number classes first.")
    print("Recommended: collect 50 to 100 samples for each number 0-9.")
    exit()

if y.value_counts().min() < 5:
    print("")
    print("Collect at least 5 samples per number class before training.")
    print("Recommended: 50 to 100 samples each for 0-9.")
    exit()

test_size_count = max(int(len(y) * 0.2), y.nunique())
test_size = test_size_count / len(y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", probability=True, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("")
print("Number Accuracy:", accuracy_score(y_test, y_pred))
print("")
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
joblib.dump(model, MODEL_FILE)

print("")
print(f"Number model saved as {MODEL_FILE}")