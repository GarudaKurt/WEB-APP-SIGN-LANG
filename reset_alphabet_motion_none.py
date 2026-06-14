import pandas as pd
import os

DATA_FILE = os.path.join("data", "alphabet", "fsl_alphabet_motion_data.csv")
BACKUP_FILE = os.path.join("data", "alphabet", "fsl_alphabet_motion_data_backup.csv")

if not os.path.exists(DATA_FILE):
    print("fsl_motion_data.csv not found.")
    exit()

df = pd.read_csv(DATA_FILE)

# Backup first
df.to_csv(BACKUP_FILE, index=False)
print(f"Backup saved as {BACKUP_FILE}")

# Clean label text
df["label"] = df["label"].astype(str).str.upper().str.strip()

before_count = len(df)
none_count = (df["label"] == "NONE").sum()

# Remove NONE rows only
df = df[df["label"] != "NONE"]

after_count = len(df)

df.to_csv(DATA_FILE, index=False)

print("DONE!")
print(f"Removed NONE samples: {none_count}")
print(f"Before: {before_count} rows")
print(f"After: {after_count} rows")
print("")
print("NONE is now reset to zero.")