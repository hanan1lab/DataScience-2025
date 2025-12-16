# --- MODEL 1: LOGISTIC REGRESSION ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. Fungsi Plot Confusion Matrix
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-User (0)', 'User (1)'],
                yticklabels=['Non-User (0)', 'User (1)'])
    plt.xlabel('Prediksi Model')
    plt.ylabel('Aktual (Kenyataan)')
    plt.title(title)
    plt.show()

# 2. Training Model
print("🚀 Training Logistic Regression...")
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train_scaled, y_train)

# 3. Prediksi ke Data Validasi
y_pred_val_lr = model_lr.predict(X_val_scaled)

# 4. Evaluasi Akurasi
acc_lr = accuracy_score(y_val, y_pred_val_lr)
print(f"✅ Model Selesai Dilatih!")
print(f"🎯 Akurasi pada Validation Set: {acc_lr:.2%}")
print("-" * 40)

# 5. Laporan Detail
print("Laporan Klasifikasi (Validation):")
print(classification_report(y_val, y_pred_val_lr))

# 6. Confusion Matrix
plot_cm(y_val, y_pred_val_lr, "Confusion Matrix - Logistic Regression (Val)")

print("\n📦 Sedang menyimpan dan mendownload model...")

# Simpan model ke file .pkl
nama_file_lr = 'model_lr.pkl'
joblib.dump(model_lr, nama_file_lr)
print(f"✅ Model berhasil disimpan sebagai: {nama_file_lr}")