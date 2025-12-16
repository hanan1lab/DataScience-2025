# --- MODEL 2: SUPPORT VECTOR MACHINE (SVM) ----
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Fungsi Plot Confusion Matrix
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

# Inisialisasi Model SVM
# - kernel='rbf':untuk data psikologi yang kompleks/non-linear
# - class_weight='balanced':Karena data Imbalanced!
print("🚀 Training SVM (Balanced Weight)...")
model_svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)

# Latih Model
model_svm.fit(X_train_scaled, y_train)

# Prediksi ke Data Validasi
y_pred_val_svm = model_svm.predict(X_val_scaled)

# Evaluasi Hasil
acc_svm = accuracy_score(y_val, y_pred_val_svm)
print(f"✅ Model SVM Selesai Dilatih!")
print(f"🎯 Akurasi SVM pada Validation Set: {acc_svm:.2%}")
print("-" * 40)

# Laporan Detail & Visualisasi
print("Laporan Klasifikasi (Validation):")
print(classification_report(y_val, y_pred_val_svm))

plot_cm(y_val, y_pred_val_svm, "Confusion Matrix - SVM Balanced (Val)")

#Simpan Model
nama_file_svm = 'model_svm.pkl'
joblib.dump(model_svm, nama_file_svm)
print(f"📦 Mengunduh {nama_file_svm}...")