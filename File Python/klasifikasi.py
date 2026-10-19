import os
import sys
import warnings

warnings.filterwarnings("ignore")

import imblearn
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "Dataset", "IBM_HR_Attrition.xlsx")
OUTPUT_DIR = os.path.normpath(os.path.join(script_dir, ".."))
MODEL_DIR = os.path.join(OUTPUT_DIR, "Model")
os.makedirs(MODEL_DIR, exist_ok=True)


def out(nama_file):
    return os.path.join(OUTPUT_DIR, nama_file)


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


judul("INPUT DATA")
df = pd.read_excel(dataset_path, sheet_name=0, engine="openpyxl")
print("Jumlah baris dan kolom:", df.shape)
print("\n5 baris pertama:")
print(df.head())
print("\nInfo dataset:")
df.info()

distribusi = df["Attrition"].value_counts()
print("\nDistribusi Attrition:")
print(distribusi)
print("\nPersentase (%):")
print((df["Attrition"].value_counts(normalize=True) * 100).round(2))

fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
sns.countplot(x="Attrition", data=df, ax=axes[0], palette="coolwarm")
axes[0].set_title("Distribusi Attrition")
axes[0].set_ylabel("Jumlah")
axes[1].pie(distribusi, labels=distribusi.index, autopct="%1.1f%%",
            colors=["#a8c6f0", "#f0b7a4"], startangle=90)
axes[1].set_title("Persentase Attrition")
plt.tight_layout()
plt.savefig(out("01_distribusi_attrition.png"), dpi=150)
plt.close()

judul("DETEKSI MISSING VALUE")
missing_per_kolom = df.isnull().sum()
print(missing_per_kolom.to_string())
total_missing = int(missing_per_kolom.sum())
print("\nTotal missing value:", total_missing)
if total_missing > 0:
    for kolom in df.columns[df.isnull().any()]:
        if df[kolom].dtype == "object":
            df[kolom] = df[kolom].fillna(df[kolom].mode()[0])
        else:
            df[kolom] = df[kolom].fillna(df[kolom].median())
    print("Missing value ditangani (median untuk numerik, modus untuk kategorikal).")
else:
    print("Tidak ada missing value, handling tidak diperlukan.")

judul("DETEKSI DUPLIKASI DATA")
duplikat = int(df.duplicated().sum())
duplikat_tanpa_id = int(df.drop(columns=["EmployeeNumber"]).duplicated().sum())
print("Jumlah baris duplikat                       :", duplikat)
print("Jumlah baris duplikat (tanpa EmployeeNumber):", duplikat_tanpa_id)
if duplikat > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Baris duplikat dihapus. Jumlah baris sekarang:", df.shape[0])
else:
    print("Tidak ada duplikasi, jumlah data tetap", df.shape[0])

judul("DETEKSI OUTLIER (METODE IQR)")
outlier_columns = [
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    "TotalWorkingYears",
    "NumCompaniesWorked",
    "YearsInCurrentRole",
    "YearsWithCurrManager",
]
baris = []
for kolom in df.select_dtypes(exclude="object").columns:
    q1, q3 = df[kolom].quantile([0.25, 0.75])
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    jumlah = int(((df[kolom] < bawah) | (df[kolom] > atas)).sum())
    if jumlah > 0:
        baris.append({
            "Kolom": kolom, "Jumlah Outlier": jumlah,
            "Batas Bawah": round(bawah, 2), "Batas Atas": round(atas, 2),
            "Ditangani": "Ya" if kolom in outlier_columns else "Tidak (diskret)",
        })
print(pd.DataFrame(baris).sort_values("Jumlah Outlier", ascending=False).to_string(index=False))

fig, axes = plt.subplots(2, 4, figsize=(10, 6))
axes = axes.flatten()
for index, kolom in enumerate(outlier_columns):
    sns.boxplot(y=df[kolom], ax=axes[index], color="#a8c6f0")
    axes[index].set_title(kolom, fontsize=9)
    axes[index].set_ylabel("")
axes[7].axis("off")
plt.suptitle("Boxplot Deteksi Outlier (Data Penuh)")
plt.tight_layout()
plt.savefig(out("02_boxplot_deteksi_outlier.png"), dpi=150)
plt.close()

judul("PENGHAPUSAN KOLOM TIDAK RELEVAN DAN ENCODING TARGET")
print("Shape sebelum drop:", df.shape)
columns_to_drop = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
for kolom in columns_to_drop:
    print("  %-15s nilai unik: %d" % (kolom, df[kolom].nunique()))
df = df.drop(columns=columns_to_drop)
print("Shape setelah drop:", df.shape)
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
print("\nEncoding target (Yes = 1, No = 0):")
print(df["Attrition"].value_counts())

judul("SPLIT DATA 80% TRAINING : 20% TESTING")
X = df.drop(columns=["Attrition"])
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_train = X_train.copy()
X_test = X_test.copy()
print("Data training:", X_train.shape[0])
print("Data testing :", X_test.shape[0])
tabel_split = pd.DataFrame(
    {
        "Total": [len(y), len(y_train), len(y_test)],
        "No (0)": [(y == 0).sum(), (y_train == 0).sum(), (y_test == 0).sum()],
        "Yes (1)": [(y == 1).sum(), (y_train == 1).sum(), (y_test == 1).sum()],
    },
    index=["Data awal", "Data training", "Data testing"],
)
tabel_split["% Yes"] = (tabel_split["Yes (1)"] / tabel_split["Total"] * 100).round(2)
print("\nDistribusi kelas:")
print(tabel_split)

judul("HANDLING OUTLIER (CAPPING IQR)")
bounds = {}
for kolom in outlier_columns:
    q1 = X_train[kolom].quantile(0.25)
    q3 = X_train[kolom].quantile(0.75)
    iqr = q3 - q1
    bounds[kolom] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


def hitung_outlier(data, kolom):
    bawah, atas = bounds[kolom]
    return int(((data[kolom] < bawah) | (data[kolom] > atas)).sum())


sebelum_train = {k: hitung_outlier(X_train, k) for k in outlier_columns}
sebelum_test = {k: hitung_outlier(X_test, k) for k in outlier_columns}

fig, axes = plt.subplots(4, 4, figsize=(10, 9.5))
for index, kolom in enumerate(outlier_columns):
    ax_sebelum = axes[index // 4, index % 4]
    sns.boxplot(y=X_train[kolom], ax=ax_sebelum, color="#a8c6f0")
    ax_sebelum.set_title(kolom + " (sebelum)", fontsize=8)
    ax_sebelum.set_ylabel("")

for kolom, (bawah, atas) in bounds.items():
    X_train[kolom] = X_train[kolom].astype(float).clip(bawah, atas)
    X_test[kolom] = X_test[kolom].astype(float).clip(bawah, atas)

for index, kolom in enumerate(outlier_columns):
    ax_sesudah = axes[2 + index // 4, index % 4]
    sns.boxplot(y=X_train[kolom], ax=ax_sesudah, color="#f0b7a4")
    ax_sesudah.set_title(kolom + " (sesudah)", fontsize=8)
    ax_sesudah.set_ylabel("")
axes[1, 3].axis("off")
axes[3, 3].axis("off")
plt.tight_layout()
plt.savefig(out("03_boxplot_handling_outlier.png"), dpi=150)
plt.close()

hasil_outlier = pd.DataFrame(
    {
        "Batas Bawah": [round(bounds[k][0], 2) for k in outlier_columns],
        "Batas Atas": [round(bounds[k][1], 2) for k in outlier_columns],
        "Outlier Train (sebelum)": [sebelum_train[k] for k in outlier_columns],
        "Outlier Train (sesudah)": [hitung_outlier(X_train, k) for k in outlier_columns],
        "Outlier Test (sebelum)": [sebelum_test[k] for k in outlier_columns],
        "Outlier Test (sesudah)": [hitung_outlier(X_test, k) for k in outlier_columns],
    },
    index=outlier_columns,
)
print(hasil_outlier.to_string())
print("\nJumlah baris training :", X_train.shape[0], "(tidak berkurang)")
print("Jumlah baris testing  :", X_test.shape[0], "(tidak berkurang)")

judul("TRANSFORMASI Z-SCORE DAN ONE-HOT ENCODING")
categorical_columns = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_columns = X_train.select_dtypes(exclude=["object"]).columns.tolist()
print("Kolom numerik     (%d):" % len(numeric_columns), numeric_columns)
print("Kolom kategorikal (%d):" % len(categorical_columns), categorical_columns)

contoh = ["Age", "MonthlyIncome", "YearsAtCompany"]
sebelum_stat = pd.DataFrame(
    {"mean train": X_train[contoh].mean(), "std train": X_train[contoh].std(ddof=0)}
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_columns),
    ]
)
X_train_transformed = preprocessor.fit_transform(X_train)  
X_test_transformed = preprocessor.transform(X_test) 
feature_names = preprocessor.get_feature_names_out()
X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
print("\nJumlah fitur setelah transformasi:", X_train_transformed.shape[1])
print("  numerik (Z-Score)  :", len(numeric_columns))
print("  one-hot (dummy)    :", X_train_transformed.shape[1] - len(numeric_columns))

kolom_z = ["numeric__" + c for c in contoh]
print("\nStatistik fitur SEBELUM Z-Score (data training):")
print(sebelum_stat.round(4))
print("\nStatistik fitur SESUDAH Z-Score:")
sesudah_stat = pd.DataFrame(
    {
        "mean train": X_train_transformed[kolom_z].mean().values,
        "std train": X_train_transformed[kolom_z].std(ddof=0).values,
        "mean test": X_test_transformed[kolom_z].mean().values,
        "std test": X_test_transformed[kolom_z].std(ddof=0).values,
    },
    index=contoh,
)
print(sesudah_stat.round(4))

judul("RESAMPLING: ROS DAN SMOTE (HANYA DATA TRAINING)")
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_ros, y_train_ros = ros.fit_resample(X_train_transformed, y_train)
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_transformed, y_train)

tabel_resampling = pd.DataFrame(
    {
        "No (0)": [(y_train == 0).sum(), (y_train_ros == 0).sum(), (y_train_smote == 0).sum()],
        "Yes (1)": [(y_train == 1).sum(), (y_train_ros == 1).sum(), (y_train_smote == 1).sum()],
    },
    index=["Baseline", "ROS", "SMOTE"],
)
tabel_resampling["Total"] = tabel_resampling.sum(axis=1)
print("Distribusi kelas data training:")
print(tabel_resampling)

fig, ax = plt.subplots(figsize=(8, 5))
tabel_resampling[["No (0)", "Yes (1)"]].plot(kind="bar", ax=ax, color=["#a8c6f0", "#f0b7a4"])
for container in ax.containers:
    ax.bar_label(container, fontsize=9)
ax.set_title("Distribusi Kelas Data Training")
ax.set_ylabel("Jumlah sampel")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(out("06_distribusi_kelas_training.png"), dpi=150)
plt.close()

data_training = {
    "Baseline": (X_train_transformed, y_train),
    "ROS": (X_train_ros, y_train_ros),
    "SMOTE": (X_train_smote, y_train_smote),
}
skenario_list = list(data_training.keys())

def buat_model(nama_model):
    if nama_model == "Naive Bayes":
        return GaussianNB()
    if nama_model == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE)
    return DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=RANDOM_STATE)


nama_model_list = ["Naive Bayes", "SVM", "Decision Tree"]
keterangan_model = {
    "Naive Bayes": "GaussianNB()",
    "SVM": "SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)",
    "Decision Tree": "DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)",
}
models = {}
for nama_model in nama_model_list:
    judul(f"TRAINING {nama_model.upper()} (3 SKENARIO)")
    print("Model:", keterangan_model[nama_model])
    for skenario in skenario_list:
        model = buat_model(nama_model)
        X_fit, y_fit = data_training[skenario]
        model.fit(X_fit, y_fit)
        models[(nama_model, skenario)] = model
        print("  %-9s dilatih dengan %d sampel" % (skenario, len(y_fit)))

judul("TESTING DAN EVALUASI (KELAS POSITIF = ATTRITION YES)")
hasil_list = []
matriks = {}
for nama_model in nama_model_list:
    print("\n" + "=" * 70)
    print("MODEL:", nama_model.upper())
    print("=" * 70)
    for skenario in skenario_list:
        model = models[(nama_model, skenario)]
        prediksi = model.predict(X_test_transformed)
        cm = confusion_matrix(y_test, prediksi)
        tn, fp, fn, tp = cm.ravel()
        hasil = {
            "Model": nama_model,
            "Skenario": skenario,
            "Akurasi": accuracy_score(y_test, prediksi),
            "Presisi": precision_score(y_test, prediksi, pos_label=1, zero_division=0),
            "Recall": recall_score(y_test, prediksi, pos_label=1),
            "F1-Score": f1_score(y_test, prediksi, pos_label=1),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        }
        hasil_list.append(hasil)
        matriks[(nama_model, skenario)] = cm
        print("\n----- %s + %s -----" % (nama_model, skenario))
        print("Confusion matrix:  TN = %d | FP = %d | FN = %d | TP = %d" % (tn, fp, fn, tp))
        print("Akurasi   : %.4f" % hasil["Akurasi"])
        print("Presisi   : %.4f" % hasil["Presisi"])
        print("Recall    : %.4f" % hasil["Recall"])
        print("F1-Score  : %.4f" % hasil["F1-Score"])
        print(classification_report(y_test, prediksi, target_names=["No (0)", "Yes (1)"], zero_division=0))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, skenario in zip(axes, skenario_list):
        ConfusionMatrixDisplay(matriks[(nama_model, skenario)], display_labels=["No", "Yes"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("%s - %s" % (nama_model, skenario), fontsize=10)
    plt.tight_layout()
    fig.savefig(out("04_confusion_matrix_" + nama_model.lower().replace(" ", "_") + ".png"), dpi=150)
    plt.close(fig)

results = pd.DataFrame(hasil_list)
judul("PERBANDINGAN 9 KOMBINASI MODEL DAN SKENARIO (DATA TESTING)")
print(results.drop(columns=["TN", "FP", "FN", "TP"]).round(4).to_string(index=False))
results.to_csv(out("hasil_perbandingan_semua_model.csv"), index=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, metrik in zip(axes.flatten(), ["Akurasi", "Presisi", "Recall", "F1-Score"]):
    tabel = results.pivot(index="Skenario", columns="Model", values=metrik).loc[skenario_list, nama_model_list]
    tabel.plot(kind="bar", ax=ax, colormap="coolwarm", legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7)
    ax.set_title(metrik, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(out("05_perbandingan_metrik_semua_model.png"), dpi=150)
plt.close()

judul("PEMILIHAN METODE TERBAIK UNTUK UI (AKURASI TERTINGGI)")
urut = results.sort_values(["Akurasi", "F1-Score"], ascending=False).reset_index(drop=True)
print("Peringkat 9 kombinasi berdasarkan akurasi (tie-break F1-Score):")
print(urut[["Model", "Skenario", "Akurasi", "Presisi", "Recall", "F1-Score"]].round(4).to_string(index=False))

ringkas = results.groupby("Model")[["Akurasi", "F1-Score"]].agg(["max", "mean"]).round(4)
print("\nRingkasan per metode (max dan rata-rata dari 3 skenario):")
print(ringkas)
metode_terbaik = urut.iloc[0]["Model"]
print("\nMetode dengan akurasi tertinggi:", metode_terbaik)

SKENARIO_UI = None
kandidat = results[results["Model"] == metode_terbaik]
if SKENARIO_UI is None:
    terbaik = kandidat.sort_values("F1-Score", ascending=False).iloc[0]
else:
    terbaik = kandidat[kandidat["Skenario"] == SKENARIO_UI].iloc[0]
print("Model untuk UI: %s + %s (akurasi %.4f, recall %.4f, F1-Score %.4f)"
      % (terbaik["Model"], terbaik["Skenario"], terbaik["Akurasi"], terbaik["Recall"], terbaik["F1-Score"]))

artefak = {
    "model": models[(terbaik["Model"], terbaik["Skenario"])],
    "preprocessor": preprocessor,
    "bounds": bounds,
    "outlier_columns": outlier_columns,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "kolom_fitur": X_train.columns.tolist(),
    "nama_model": terbaik["Model"],
    "skenario": terbaik["Skenario"],
    "metrik": {k: float(terbaik[k]) for k in ["Akurasi", "Presisi", "Recall", "F1-Score"]},
    "default_numerik": {c: float(X_train[c].median()) for c in numeric_columns},
    "rentang_numerik": {c: (float(X[c].min()), float(X[c].max())) for c in numeric_columns},
    "opsi_kategorik": {c: sorted(X[c].unique().tolist()) for c in categorical_columns},
    "default_kategorik": {c: X_train[c].mode()[0] for c in categorical_columns},
    "nilai_diskret": {c: sorted(X[c].unique().tolist()) for c in numeric_columns if X[c].nunique() <= 6},
}
joblib.dump(artefak, os.path.join(MODEL_DIR, "model_terbaik.joblib"))
