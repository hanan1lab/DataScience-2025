# --- SPLITTING DATA ---

# Tahap 1: Ambil 70% untuk Training
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Tahap 2: Pecah sisa 30% tadi menjadi dua bagian sama besar (Validation & Test)
# 50% dari 30% total = 15% total data asli
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("✅ Data berhasil dibagi.")
print(f"Training Set   (70%) : {X_train.shape[0]} baris")
print(f"Validation Set (15%) : {X_val.shape[0]} baris")
print(f"Test Set       (15%) : {X_test.shape[0]} baris")


