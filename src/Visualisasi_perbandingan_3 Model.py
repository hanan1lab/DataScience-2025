# --- KOMPARASI FINAL: TABEL & VISUALISASI ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("🚀 Sedang membandingkan performa 3 model pada DATA TEST...")


y_pred_lr = model_lr.predict(X_test_scaled)

y_pred_svm = model_svm.predict(X_test_scaled)

y_prob_dl = model_dl.predict(X_test_scaled, verbose=0)
y_pred_dl = (y_prob_dl > 0.5).astype("int32").flatten()


def get_score(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (User)': precision_score(y_true, y_pred, pos_label=1),
        'Recall (User)': recall_score(y_true, y_pred, pos_label=1),
        'F1-Score': f1_score(y_true, y_pred, average='macro')
    }


ranking = []
ranking.append(get_score(y_test, y_pred_lr, 'Logistic Regression'))
ranking.append(get_score(y_test, y_pred_svm, 'SVM (Balanced)'))
ranking.append(get_score(y_test, y_pred_dl, 'Deep Learning (MLP)'))

# Buat DataFrame (Tabel)
df_ranking = pd.DataFrame(ranking)

print("\n🏆 TABEL PERBANDINGAN MODEL (DATA TEST):")
print("-" * 100)

print(df_ranking.round(4).to_markdown(index=False))
print("-" * 100)

plt.figure(figsize=(12, 6))


df_melt = df_ranking.melt(id_vars="Model", var_name="Metrik", value_name="Skor")


ax = sns.barplot(data=df_melt, x="Model", y="Skor", hue="Metrik", palette="viridis")

plt.title("Perbandingan Performa Model (Data Test)", fontsize=14, fontweight='bold')
plt.ylim(0.4, 1.05)
plt.ylabel("Skor (0.0 - 1.0)")
plt.xlabel("")
plt.legend(loc='lower right', title='Metrik')
plt.grid(axis='y', linestyle='--', alpha=0.3)


for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

plt.tight_layout()
plt.show()