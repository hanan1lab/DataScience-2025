# --- MODEL 3: DEEP LEARNING ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
from sklearn.metrics import confusion_matrix, classification_report

#Konfigurasi Arsitektur
model_dl = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🚀 Mulai Training Model Deep Learning (20 Epochs)...")
start_time = time.time()

#Training
history = model_dl.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=20,
    batch_size=16,
    verbose=1
)

end_time = time.time()
print(f"✅ Training Selesai dalam {end_time - start_time:.2f} detik!")

# Visualisasi Grafik (Loss & Accuracy)
list_epoch = range(1, len(history.history['accuracy']) + 1)
plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(list_epoch, history.history['accuracy'], label='Train Acc', marker='.')
plt.plot(list_epoch, history.history['val_accuracy'], label='Val Acc', marker='.')
plt.title('Model Accuracy (Per Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(list_epoch, history.history['loss'], label='Train Loss', marker='.')
plt.plot(list_epoch, history.history['val_loss'], label='Val Loss', marker='.')
plt.title('Model Loss (Per Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# Prediksi ke Data Validasi (Untuk Evaluasi)
y_pred_prob_val = model_dl.predict(X_val_scaled, verbose=0)
y_pred_val_dl = (y_pred_prob_val > 0.5).astype("int32")

# Tampilkan Confusion Matrix Validation
cm_val = confusion_matrix(y_val, y_pred_val_dl)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-User (0)', 'User (1)'],
            yticklabels=['Non-User (0)', 'User (1)'])
plt.xlabel('Prediksi Model')
plt.ylabel('Aktual (Kenyataan)')
plt.title('Confusion Matrix - Deep Learning (Validation)')
plt.show()

# Print Akurasi Validation
_, acc_dl_val = model_dl.evaluate(X_val_scaled, y_val, verbose=0)
print(f"🎯 Akurasi Validation: {acc_dl_val:.2%}")
print("-" * 40)

# Classification Report
print("Laporan Klasifikasi Lengkap (Validation Data):")
print(classification_report(y_val, y_pred_val_dl))

#Simpan Model
nama_file_dl = 'model_dl.h5'
model_dl.save(nama_file_dl) # Simpan
print(f"📦 Mengunduh {nama_file_dl}...")

#cek summary model dl
model_dl.summary()

# --- EVALUASI FINAL (DATA TEST) - LENGKAP ---
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("🚀 Memulai Evaluasi Final pada Data Test...")

# 1. Prediksi menggunakan Data Test
y_pred_prob = model_dl.predict(X_test_scaled)

# 2. Ubah Probabilitas menjadi Kelas (0 atau 1)
y_pred_test = (y_pred_prob > 0.5).astype("int32").flatten()

# 3. Hitung Akurasi Final
acc_test = accuracy_score(y_test, y_pred_test)
print(f"\n🏆 HASIL AKHIR PADA DATA TEST:")
print(f"🎯 Akurasi Final: {acc_test:.2%}")
print("-" * 40)

# 4. Tampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-User (0)', 'User (1)'],
            yticklabels=['Non-User (0)', 'User (1)'])
plt.xlabel('Prediksi Model')
plt.ylabel('Kenyataan (Aktual)')
plt.title('Confusion Matrix - Deep Learning (Test Data)')
plt.show()

# Laporan Detail
print("\nLaporan Klasifikasi Lengkap:")
print(classification_report(y_test, y_pred_test))

print("-" * 40)
print("🔍 SAMPEL HASIL PREDIKSI (10 Data Pertama):")

# Buat DataFrame Perbandingan
df_hasil = pd.DataFrame({
    'Actual (Kenyataan)': y_test.values if hasattr(y_test, 'values') else y_test,
    'Predicted (Model)': y_pred_test,
    'Probabilitas': y_pred_prob.flatten()
})

# Tambahkan kolom Status (Benar/Salah)
df_hasil['Status'] = df_hasil.apply(
    lambda x: '✅ Benar' if x['Actual (Kenyataan)'] == x['Predicted (Model)'] else '❌ Salah', axis=1
)

# Tampilkan 10 baris pertama
print(df_hasil.head(10).to_markdown(index=False))