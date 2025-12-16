!pip install ucimlrepo pandas --quiet
import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from google.colab import files

# Mengambil dataset dari UCI
drug = fetch_ucirepo(id=373)

# Memisahkan fitur dan target (label)
X = drug.data.features
y = drug.data.targets

# Membuat DataFrame lengkap (fitur + target)
df = pd.concat([X, y], axis=1)

df