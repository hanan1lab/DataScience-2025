# 1. Definisi Daftar Kolom Fitur (Input)
feature_cols = [
    'age', 'gender',
    'nscore', 'escore', 'oscore', 'ascore', 'cscore',
    'impuslive', 'ss']

# 2. Definisi Kolom Target
target_col = 'cannabis'

# 3. Buat DataFrame Baru (df_selected)
df_selected = df[feature_cols + [target_col]].copy()

# 4. Tampilkan Hasil
print("=== Kolom yang digunakan ===")
print(f"Jumlah Fitur : {len(feature_cols)} kolom")
print(f"Nama Fitur   : {feature_cols}")
print(f"Target       : {target_col}")
print(f"Ukuran Data  : {df_selected.shape}")

print("\n=== PREVIEW DATA BARU (df_selected) ===")
df_selected.head()

# --- PREPROCESSING Encoding ---
# 1. Ubah isi kolom 'cannabis' langsung menjadi angka
# Jika 'CL0' -> 0 (Non-User), selain itu -> 1 (User)
df_selected['cannabis'] = df_selected['cannabis'].apply(lambda x: 0 if x == 'CL0' else 1)

# 2. Pisahkan Fitur dan Target
# X = Semua kolom KECUALI 'cannabis'
X = df_selected.drop(columns=['cannabis'])

# y = Kolom 'cannabis' saja
y = df_selected['cannabis']

print("-" * 30)
print(" Data Target (y):")
print(y.head())
print("\nDistribusi:")
print(y.value_counts())

#Simpan Data Hasil Preprocessing ke CSV
# gabungkan lagi X dan y sebentar untuk disimpan
df_final = X.copy()
df_final['target'] = y

nama_file_baru = 'data_processed.csv'
df_final.to_csv(nama_file_baru, index=False)

print(f"\n✅ Data berhasil disimpan ke: {nama_file_baru}")

df_final.head()