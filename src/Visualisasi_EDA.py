# --- VISUALISASI EDA  ---

df_viz = X.copy()
df_viz['target'] = y['cannabis']

# 2. Definisi Mapping
age_map = {
    -0.95197: '18-24',
    -0.07854: '25-34',
    0.49788:  '35-44',
    1.09449:  '45-54',
    1.82213:  '55-64',
    2.59171:  '65+'
}

gender_map = {
    0.48246: 'Female',
    -0.48246: 'Male'
}

# 3. Terapkan Mapping ke Kolom Baru
df_viz['label_age'] = df_viz['age'].map(age_map)
df_viz['label_gender'] = df_viz['gender'].map(gender_map)
df_viz['label_cannabis'] = df_viz['target'].apply(lambda x: 'User' if x == 'CL0' else 'Non-User')

plt.figure(figsize=(18, 5))

# Grafik 1: Total Distribusi Pengguna
plt.subplot(1, 3, 1)
sns.countplot(x='label_cannabis', data=df_viz, palette='viridis')
plt.title('1. Total Distribusi User vs Non-User (Data Full)')
plt.xlabel('')
plt.ylabel('Jumlah Responden')

# Grafik 2: Penggunaan Berdasarkan Umur
plt.subplot(1, 3, 2)
order_age = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
sns.countplot(x='label_age', hue='label_cannabis', data=df_viz, order=order_age, palette='magma')
plt.title('2. Sebaran Umur')
plt.xlabel('Umur')

# Grafik 3: Skor Neuroticism
plt.subplot(1, 3, 3)
sns.boxplot(x='label_cannabis', y='nscore', data=df_viz, palette='coolwarm')
plt.title('3. Neuroticism Score ')

plt.tight_layout()
plt.show()