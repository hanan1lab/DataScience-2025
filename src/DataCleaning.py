#cek info data
df.info()

#cek banyak data
len(df)

#Menampilkan 5 data awal 
df.head()

#Melihat Missing Value
df.isnull().sum()

# --- CEK DATA DUPLIKAT ---

# 1. Hitung jumlah baris yang duplikat
jumlah_duplikat = df.duplicated().sum()

print(f"Jumlah Data Duplikat: {jumlah_duplikat}")

# 2. (Opsional) Tampilkan data yang duplikat jika ada
if jumlah_duplikat > 0:
    print("\nContoh data yang duplikat:")
    print(df[df.duplicated()].head())
else:
    print("✅ Data aman (Clean), tidak ada duplikat.")

#Menampilkan tipe data dari data frame
df.dtypes

# --- CEK NOISE / OUTLIERS ---

# Titik-titik hitam di luar garis "kumis" adalah NOISE (Outliers)
plt.figure(figsize=(15, 6))
cols_to_check = ['nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impuslive', 'ss']
sns.boxplot(data=df[cols_to_check])
plt.title("Deteksi Noise (Outliers) pada Fitur Psikologis")
plt.grid(True, alpha=0.3)
plt.show()

# --- CEK DISTRIBUSI AWAL (RAW DATA) ---

# 1. Tampilkan Grafik Data Asli (CL0 - CL6)
plt.figure(figsize=(8, 5))
urutan = sorted(df['cannabis'].unique())
sns.countplot(x=df['cannabis'], order=urutan, palette='viridis')

plt.title('Distribusi Detail Penggunaan Cannabis (Data Mentah)')
plt.xlabel('Tingkat Penggunaan (CL0 = Never, CL6 = Heavy)')
plt.ylabel('Jumlah Responden')
plt.show()

# 2. Hitung Imbalance .
non_user_count = len(df[df['cannabis'] == 'CL0'])
user_count = len(df) - non_user_count

print("-" * 40)
print("📊 ANALISIS KETIMPANGAN (UNTUK MODELING):")
print(f"Total Data      : {len(df)}")
print(f"Non-User (CL0)  : {non_user_count} orang (Akan jadi Label 0)")
print(f"User (CL1-CL6)  : {user_count} orang (Akan jadi Label 1)")

ratio = user_count / non_user_count
print(f"\n⚠️ Kesimpulan: Data IMBALANCED dengan rasio {ratio:.1f} : 1")
print("   (User jauh lebih banyak daripada Non-User)")
print("-" * 40)

