import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import datetime

# Log dosyasını temizle ve başlık ekle
with open("algorithms_result.txt", "w", encoding="utf-8") as log_file:
    log_file.write(f"=== Safra Taşı Veri Seti Algoritma Sonuçları ===\n")
    log_file.write(f"Çalıştırma Tarihi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# Veriyi oku
data = pd.read_csv("dataset-uci.csv")  # Dosya yolunu projeye göre ayarla

# Eksik veri doldurma
imputer = SimpleImputer(strategy="most_frequent")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Kategorik verileri etiketle
le = LabelEncoder()
categorical_cols = data_imputed.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data_imputed[col] = le.fit_transform(data_imputed[col])

# Bağımlı ve bağımsız değişkenleri ayır
X = data_imputed.drop("Gallstone Status", axis=1)
y = data_imputed["Gallstone Status"]

# Eğitim/test verisi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standartlaştırma (kNN için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Değerlendirme fonksiyonu
def evaluate_model(name, y_true, y_pred):
    # Ekrana yazdır
    print(f"\n{name} Sonuçları:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
      # Log dosyasına kaydet
    with open("algorithms_result.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{name} Sonuçları:\n")
        log_file.write(f"Accuracy: {accuracy_score(y_true, y_pred)}\n")
        log_file.write(f"Precision: {precision_score(y_true, y_pred)}\n")
        log_file.write(f"Recall: {recall_score(y_true, y_pred)}\n")
        log_file.write(f"F1 Score: {f1_score(y_true, y_pred)}\n")
        log_file.write(f"\nClassification Report:\n{classification_report(y_true, y_pred)}\n")

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
evaluate_model("Naive Bayes", y_test, y_pred_nb)

# kNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
evaluate_model("kNN", y_test, y_pred_knn)

# Karar Ağacı
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
evaluate_model("Karar Ağacı", y_test, y_pred_dt)

# Karar ağacı görselleştir
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Karar Ağacı Görselleştirmesi")
plt.savefig("figures/decision_tree.png")
plt.show()

# Log dosyasına özet bilgileri ekle
with open("algorithms_result.txt", "a", encoding="utf-8") as log_file:
    log_file.write("\n" + "="*50 + "\n")
    log_file.write("ÖZET KARŞILAŞTIRMA\n")
    log_file.write("="*50 + "\n")
    log_file.write(f"Veri seti boyutu: {data.shape}\n")
    log_file.write(f"Eğitim seti boyutu: {X_train.shape}\n")
    log_file.write(f"Test seti boyutu: {X_test.shape}\n\n")
    log_file.write("En iyi performans gösteren model:\n")
    
    # En iyi modeli bul
    acc_nb = accuracy_score(y_test, y_pred_nb)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    
    if acc_nb >= acc_knn and acc_nb >= acc_dt:
        best_model = "Naive Bayes"
        best_acc = acc_nb
    elif acc_knn >= acc_nb and acc_knn >= acc_dt:
        best_model = "kNN"
        best_acc = acc_knn
    else:
        best_model = "Karar Ağacı"
        best_acc = acc_dt
    
    log_file.write(f"{best_model} (Accuracy: {best_acc:.4f})\n\n")
    
print("\nAlgoritma sonuçları 'algorithms_result.txt' dosyasına kaydedildi.")
