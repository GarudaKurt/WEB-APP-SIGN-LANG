import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = os.path.join("data", "alphabet", "fsl_alphabet_data.csv")
MODEL_FILE = os.path.join("models", "alphabet", "fsl_alphabet_model.joblib")

if not os.path.exists(DATA_FILE):
    print("No fsl_data.csv found. Run collect_data.py first.")
    exit()

try:
    df = pd.read_csv(DATA_FILE, encoding="utf-8", encoding_errors="replace")
except Exception as e:
    print("Error reading fsl_data.csv:")
    print(e)
    exit()

if df.empty:
    print("Dataset is empty. Collect samples first.")
    exit()

df["label"] = df["label"].astype(str).str.upper().str.strip()
df = df[df["label"].str.match(r"^[A-Z]$", na=False)]

print("Samples per letter:")
print(df["label"].value_counts().sort_index())

X = df.drop(columns=["label"])
y = df["label"]

if y.nunique() < 2:
    print("")
    print("You only collected one letter.")
    print("Collect at least 2 letters first, example A and B.")
    print("Then run train_model.py again.")
    exit()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", probability=True))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_FILE)

print("")
print(f"Model saved as {MODEL_FILE}")