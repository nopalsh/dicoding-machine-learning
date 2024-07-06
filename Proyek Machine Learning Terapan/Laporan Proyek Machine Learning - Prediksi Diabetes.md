# Laporan Proyek Machine Learning - Prediksi Diabetes

## Domain Proyek

Diabetes merupakan masalah kesehatan global yang memengaruhi jutaan orang di seluruh dunia. Prediksi keberadaan diabetes pada pasien berdasarkan faktor risiko klinis dapat membantu dalam deteksi dini dan pengelolaan penyakit ini.

## Business Understanding

### Problem Statements

1. Bagaimana kita dapat memprediksi keberadaan diabetes pada pasien berdasarkan faktor-faktor klinis yang terkait?
2. Bagaimana kita dapat meningkatkan deteksi dini diabetes untuk pengelolaan yang lebih efektif?

### Goals

1. Membangun model machine learning untuk memprediksi keberadaan diabetes pada pasien.
2. Meningkatkan deteksi dini diabetes dengan meningkatkan akurasi prediksi model.

## Data Understanding

Data yang digunakan berasal dari dataset "diabetes.csv". Dataset ini terdiri dari 768 entri dengan 9 kolom yang mencakup informasi klinis tentang pasien, termasuk keberadaan diabetes sebagai target.
Link Datasets : https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

### Variabel pada Dataset:
- Pregnancies: Jumlah kehamilan
- Glucose: Kadar glukosa plasmatik 2 jam dalam tes toleransi glukosa
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit trisep (mm)
- Insulin: Kadar insulin serum 2 jam (mu U/ml)
- BMI: Indeks massa tubuh (berat dalam kg / (tinggi dalam m) ** 2)
- DiabetesPedigreeFunction: Riwayat keluarga diabetes
- Age: Usia (tahun)
- Outcome: Keberadaan diabetes (0: Tidak, 1: Ya)

## Data Preparation

Pada tahap ini, dilakukan pemisahan data menjadi data latih dan data uji dengan rasio 80:20 menggunakan train_test_split.

## Modeling

Model K-Nearest Neighbors (KNN) digunakan untuk memprediksi keberadaan diabetes. Model ini diinisialisasi dengan 3 tetangga dan dilatih menggunakan data latih.

## Evaluation

Metrik evaluasi yang digunakan adalah akurasi. Model KNN memberikan akurasi sekitar 64.93% dalam memprediksi keberadaan diabetes pada data uji.
