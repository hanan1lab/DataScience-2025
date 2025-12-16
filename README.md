# 📘 Judul Proyek
Klasifikasi Pengguna Obat (Cannabis) berdasarkan Profil Psikologis Menggunakan Deep Learning

## 👤 Informasi
- **Nama:** Hanan Labib Rasyaddin
- **NIM:** 234311041
- **Repo:** https://github.com/hanan1lab/DataScience-2025 
- **Video:** https://youtu.be/CUJqSck792A

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan deteksi dini risiko penyalahgunaan narkoba berbasis data psikologis (*psychometric data*).
- Melakukan data preparation meliputi cleaning, binary encoding, dan scaling menggunakan **StandardScaler**.
- Membangun 3 model: **Baseline (Logistic Regression)**, **Advanced (SVM Balanced)**, **Deep Learning (MLP)**.
- Melakukan evaluasi menggunakan metrik **Accuracy, F1-Score, dan Recall** untuk menangani dataset yang tidak seimbang (*imbalanced*).

---

# 2. 📄 Problem & Goals
**Problem Statements:**
- Penyalahgunaan narkoba sulit dideteksi secara dini tanpa metode klinis yang mahal.
- Dataset memiliki karakteristik *imbalanced* (User > Non-User), yang menyebabkan model cenderung bias ke kelas mayoritas.
- Hubungan antara sifat kepribadian (*personality traits*) dan perilaku risiko bersifat kompleks dan non-linear.

**Goals:**
- Membangun model klasifikasi biner dengan target akurasi > 80%.
- Menangani ketimpangan data menggunakan teknik *Class Weighting*.
- Membandingkan performa model linear, machine learning klasik, dan neural network.

---
## 📁 Struktur Folder
```
project/
│
├── data/
│   └── data_processed.csv
│   └── drug_consumption.data
|
├── images/
│   └── Cek Noise Outlier.png
│   └── Confusion Matrix LR.png
│   └── Confusion matrix DL pada data test.png
│   └── Confusion matrix DL.png
|   └── Distribusi data targer.png
|   └── Visualisasi accuracy dan loss per epoch dl.png
|   └── Visualisasi Eda.png
|   └──Visualisasi perbandingan model.png
|   └── Confusion matrix SVM.png
|
├── models/
│   ├── model_dl.h5
│   ├── model_lr.pkl
│   └── model_svm.pkl
│
├── notebooks/
│   └── 234311041_Hanan_Labib_Rasyaddin_UAS_Data_Science.ipynb
│
├── src/
│   └── DataCleaning.py
│   └── Feature_Engineering.py
│   └── Import dan Load dataset.py
│   └── Model_DeepLearning_MLP.py
│   └── Model_LogisticRegression.py
│   └── Model_SVM.py
│   └── Normalisasi.py
│   └── Splittingdata.py
|   └── Visualisasi_EDA.py
|   └── Visualisasi_perbandingan_3 Model.py
│
├── .gitignore
├── Laporan Proyek Machine Learning.pdf
├── Checklist Submit.md
├── LICENSE
├── README.md
└── requirements.txt
```

---
# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository / Figshare
- **Jumlah Data:** 1885 Baris, 12 Fitur Utama
- **Tipe:** Tabular (Kuantitatif & Kategorikal yang sudah dikuantifikasi)

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
|Demografi | Age, Gender.|
| Personality Scores | Nscore (Neuroticism), Escore (Extraversion), Oscore (Openness). |
| Risk Traits | Impulsive (Impulsivitas), SS (Sensation Seeking) |
| Target (Class) | Label: 'User' (1) atau 'Non-User' (0) |

---

# 4. 🔧 Data Preparation
- **Cleaning:** Pengecekan missing values (Data bersih 100%).
- **Transformation:** Encoding target menjadi biner (User vs Non-User) dan Feature Scaling (StandardScaler).
- **Splitting:** Stratified Split (70% Train, 15% Val, 15% Test).
- **Handling Imbalance:** Menggunakan parameter `class_weight='balanced'` pada model SVM.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** **Logistic Regression** (Linear model, simple & fast).
- **Model 2 – Advanced ML:** **Support Vector Machine (SVM)** (Kernel RBF, Class Weight Balanced).
- **Model 3 – Deep Learning:** **Multilayer Perceptron (MLP)** dengan arsitektur: Input(12) -> Dense(16, ReLU) -> Dense(8, ReLU) -> Output(1, Sigmoid).

---

# 6. 🧪 Evaluation
**Metrik:** **F1-Score (Macro)** & Accuracy. (F1-Score penting karena data imbalanced).

### Hasil Singkat
| Model | Accuracy | F1-Score | Catatan |
|-------|--------|---------|---------|
| Baseline (LogReg) | 81.27% | 0.66 | Cepat, namun Recall untuk Non-User rendah. |
| Advanced (SVM) | 75.97% | 0.71 | Recall paling tinggi (sensitif), tapi banyak False Positive. |
| Deep Learning (MLP) | **80.92%** | **0.68** | **Model Terbaik.** Seimbang antara Akurasi dan deteksi kelas minoritas. |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Deep Learning (Multilayer Perceptron).
- **Alasan:** Menunjukkan kurva pembelajaran (*learning curve*) yang stabil (Good Fit) dan mampu menangkap pola non-linear dari fitur kepribadian.
- **Insight penting:** Fitur **Sensation Seeking (SS)** dan **Openness** adalah indikator terkuat dalam memprediksi risiko penggunaan obat.

---

# 8. 🔮 Future Work
- [x] Hyperparameter tuning lebih ekstensif
- [x] Ensemble methods (combining models)
- [ ] Menambah variasi data responden dari negara lain
- [ ] Deployment (Streamlit/FastAPI)

---

# 9. 🔁 Reproducibility
Untuk menjalankan proyek ini di lokal, gunakan environment berikut:
Clone Repository:
git clone https://github.com/hanan1lab/DataScience-2025
cd DataScience-2025
Install Dependencies:
pip install -r requirements.txt

Jalankan Notebook: Buka file di notebooks/234311041_Hanan_Labib_Rasyaddin_UAS_Data_Science.ipynb menggunakan Jupyter Notebook atau VS Code.

Gunakan environment:
**Python 3.10+**
Libraries utama:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` (Keras)
- `seaborn`
- `joblib`

Instalasi:
```bash
pip install -r requirements.txt
