import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

PHRASE_DATA_FILE = os.path.join("data", "phrases", "fsl_phrase_data.csv")
PHRASE_MODEL_FILE = os.path.join("models", "phrases", "fsl_phrase_model.joblib")

SEQUENCE_LENGTH = 30
MAX_HANDS = 2
EXPECTED_FEATURE_COUNT = SEQUENCE_LENGTH * MAX_HANDS * 21 * 3
EXPECTED_COLUMN_COUNT = 1 + EXPECTED_FEATURE_COUNT

if not os.path.exists(PHRASE_DATA_FILE):
    print("No phrase dataset found.")
    print("Run collect_phrase_data.py first.")
    exit()

try:
    df = pd.read_csv(PHRASE_DATA_FILE, encoding="utf-8", encoding_errors="replace")
except Exception as e:
    print("Error reading phrase dataset:")
    print(e)
    exit()

if df.empty:
    print("Phrase dataset is empty. Collect phrase samples first.")
    exit()

if len(df.columns) != EXPECTED_COLUMN_COUNT:
    print("Invalid phrase dataset format.")
    print(f"Expected {EXPECTED_COLUMN_COUNT} columns, but found {len(df.columns)}.")
    print("")
    print("This usually means your CSV was collected using the old 1-hand phrase collector.")
    print("Back up and delete data/phrases/fsl_phrase_data.csv, then recollect phrase data.")
    exit()

df["label"] = df["label"].astype(str).str.upper().str.strip()
df["label"] = df["label"].str.replace(" ", "_", regex=False)
df["label"] = df["label"].str.replace("?", "", regex=False)

df = df[df["label"].notna()]
df = df[df["label"] != ""]
df = df[df["label"] != "NAN"]

if df.empty:
    print("No valid phrase labels found.")
    exit()

X = df.drop(columns=["label"])
X = X.apply(pd.to_numeric, errors="coerce")

valid_rows = X.notna().all(axis=1)

df = df.loc[valid_rows].copy()
X = X.loc[valid_rows].copy()
y = df["label"]

if df.empty:
    print("No valid numeric feature rows found.")
    exit()

print("Samples per phrase:")
print(y.value_counts().sort_index())

if y.nunique() < 2:
    print("")
    print("Collect at least 2 classes first.")
    print("Example: one phrase and NONE.")
    exit()

if y.value_counts().min() < 5:
    print("")
    print("Collect at least 5 samples per phrase class.")
    print("Recommended: 50 to 100 samples each, including NONE.")
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
print("Phrase Accuracy:", accuracy_score(y_test, y_pred))
print("")
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(PHRASE_MODEL_FILE), exist_ok=True)
joblib.dump(model, PHRASE_MODEL_FILE)

print("")
print(f"Phrase model saved as {PHRASE_MODEL_FILE}")