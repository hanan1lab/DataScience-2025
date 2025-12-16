# --- NORMALISASI FITUR (StandardScaler) ---

# 1. Inisialisasi Scaler
scaler = StandardScaler()

# 2. FIT hanya pada data TRAIN (Agar tidak curang/data leakage)
scaler.fit(X_train)

# 3. TRANSFORM semua data (Train, Val, Test) menggunakan rumus dari Train
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print("✅ Normalisasi Selesai!")
print("Data sekarang tersimpan di variabel:")
print("   - X_train_scaled, y_train")
print("   - X_val_scaled,   y_val")
print("   - X_test_scaled,  y_test")

print("\nContoh data setelah dinormalisasi (Lihat, angkanya jadi sekitar -2 s/d 2):")
print(X_train_scaled.head())