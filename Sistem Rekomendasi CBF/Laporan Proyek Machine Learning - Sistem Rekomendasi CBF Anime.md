# Laporan Proyek Machine Learning - Mochamad Naufal Shofy
# Project Overview

## Domain Proyek
 Anime telah menjadi bagian tak terpisahkan dari Jepang dan mendapatkan pengakuan global^[1]^. Dalam karya ini, fokus pada pengembangan sistem rekomendasi anime bertujuan untuk menyediakan konten yang relevan kepada penonton, mengingat popularitas anime yang terus berkembang. Proyek ini menjadi penting karena fenomena budaya anime telah mencapai tingkat global^[1]^, dan kebutuhan akan rekomendasi konten yang sesuai dengan preferensi penonton semakin meningkat. Teknologi Machine Learning membantu dalam menganalisis pola dan kesamaan antara anime berdasarkan fitur-fitur yang ada, seperti genre dan rating. Dengan memanfaatkan algoritma Machine Learning, sistem rekomendasi dapat menyajikan rekomendasi yang lebih akurat dan personal kepada penonton. Dengan demikian, pengguna dapat dengan mudah menemukan anime yang sesuai dengan preferensi mereka.

# Business Understanding
Sistem rekomendasi yang sesuai dengan preferensi pengguna merupakan sistem yang mampu memahami preferensi individu pengguna berdasarkan data yang ada, seperti genre dan rating, serta dapat memberikan rekomendasi yang paling relevan dan menarik bagi pengguna. 
Kemampuan memberikan rekomendasi anime yang sesuai dengan preferensi pengguna secara akurat adalah kunci keberhasilan proyek ini, karena dapat meningkatkan keterlibatan pengguna, memperluas basis pengguna, dan meningkatkan retensi pengguna
### Problem Statements:
#####  Pernyataan Masalah 1:
Bagaimana memberikan rekomendasi anime yang relevan kepada pengguna berdasarkan preferensi mereka?
### Goals:
#####  Jawaban Pernyataan Masalah 1:
Mengembangkan sistem rekomendasi anime yang menggunakan teknologi Machine Learning dengan pendekatan Content Based Filtering  untuk menganalisis fitur-fitur anime dan memberikan rekomendasi yang kepada pengguna.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari Kaggle dengan judul "Anime Recommendation Database 2020".  Dapat mengunduh dataset tersebut dari tautan berikut: [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020?select=anime.csv).

DataFrame awal memiliki 17.562 entri dengan 35 kolom yang mencakup informasi tentang anime. Informasi mengenai variabel atau fitur pada DataFrame tersebut adalah sebagai berikut:

1. **MAL_ID**: ID unik untuk setiap anime
2. **Name**: Nama anime
3. **Score**: Skor atau peringkat anime.
4. **Genres**: Genre-genre yang terkait dengan anime tersebut
5. **English name**: Nama anime dalam bahasa Inggris
6. **Japanese name**: Nama anime dalam bahasa Jepang
7. **Type**: Jenis anime (TV, Movie, OVA, Special, ONA, Music, Unknown)
8. **Episodes**: Jumlah episode anime
9. **Aired**: Tanggal tayang anime
10. **Premiered**: Tanggal premier anime
11. **Producers**: Produser anime
12. **Licensors**: Pemegang lisensi anime
13. **Studios**: Studio produksi anime
14. **Source**: Sumber asal anime (manga, novel, dll.)
15. **Duration**: Durasi anime
16. **Rating**: Rating konten anime (G - All Ages, PG - Children, PG-13 - Teens 13 or older, R - 17+ (violence & profanity), R+ - Mild Nudity, Unknown)
17. **Ranked**: Peringkat anime
18. **Popularity**: Popularitas anime
19. **Members**: Jumlah anggota yang menambahkan anime ke daftar tonton mereka
20. **Favorites**: Jumlah kali anime ditambahkan ke daftar favorit
21. **Watching**: Jumlah orang yang sedang menonton anime
22. **Completed**: Jumlah orang yang telah menyelesaikan menonton anime
23. **On-Hold**: Jumlah orang yang menunda menonton anime
24. **Dropped**: Jumlah orang yang berhenti menonton anime
25. **Plan to Watch**: Jumlah orang yang berencana menonton anime
26. **Score-10** hingga **Score-1**: Jumlah peringkat untuk setiap skor dari 1 hingga 10

DataFrame kedua (df_filtered) DataFrame sebelumnya yang sudah di pre-processing dan filter dari memiliki 16.217 entri dengan 52 kolom yang juga mencakup informasi tentang anime. Informasi mengenai variabel atau fitur pada DataFrame tersebut adalah sebagai berikut:

1. **MAL_ID**: ID unik untuk setiap anime
2. **Name**: Nama anime
3. **Score**: Skor atau peringkat anime
4. **Genres**: Genre-genre yang terkait dengan anime tersebut
5. **Type**: Jenis anime (TV, Movie, OVA, Special, ONA, Music, Unknown)
6. **Aired**: Tanggal tayang anime
7. **Rating**: Rating konten anime (G - All Ages, PG - Children, PG-13 - Teens 13 or older, R - 17+ (violence & profanity), R+ - Mild Nudity, Unknown)
8. **Members**: Jumlah anggota yang menambahkan anime ke daftar tonton mereka
9. Dan variabel atau fitur pada genre dalam gambar berikut:
Gambar 1.  Distribusi Genre Anime
![Gambar 1.  Distribusi Genre Anime](https://i.ibb.co/8B43Bxs/download.png)
Gambar 1 menunjukkan distribusi genre anime, di mana setiap bar mewakili jumlah anime yang termasuk dalam genre tersebut.

## Data Preparation

Pada tahap ini, dilakukan berbagai teknik untuk mempersiapkan data sebelum dilakukan analisis dan pemodelan. Berikut adalah teknik yang digunakan:

### 1. Seleksi Fitur
- Hanya kolom yang dianggap relevan dipertahankan untuk analisis lebih lanjut, seperti 'MAL_ID', 'Name', 'Score', 'Genres', 'Type', 'Aired', 'Rating', dan 'Members'.
- Seleksi fitur dilakukan untuk mengurangi dimensi data dan meningkatkan kinerja model. Dengan memilih hanya fitur-fitur yang paling relevan, waktu komputasi dapat dikurangi.
     
### 2. Pengkodean One-Hot untuk Genre
- Kolom 'Genres' yang berisi informasi genre anime dibagi menjadi kolom terpisah dengan menerapkan teknik pengkodean one-hot.
- Pengkodean one-hot untuk genre diperlukan karena kolom 'Genres' berisi informasi tentang genre-genre yang dimiliki oleh setiap anime. Dengan membagi genre menjadi kolom terpisah, dapat memperlakukan setiap genre sebagai fitur yang independen, memungkinkan algoritma untuk memahami hubungan antara genre tertentu dan preferensi pengguna.
   
### 3. Konversi Nilai Kategorikal
- Nilai kategorikal pada kolom 'Rating' dan 'Type' dikonversi menjadi nilai numerik menggunakan peta pemetaan.
- Konversi nilai kategorikal menjadi nilai numerik diperlukan agar algoritma machine learning dapat memahami dan mengolah informasi tersebut. Ini membantu meningkatkan kinerja model dengan memungkinkan algoritma untuk memproses data kategorikal sebagai masukan.
   
### 4. Penanganan Data Missing
- Setelah transformasi, dilakukan pengecekan untuk nilai yang hilang. Dalam kasus ini, nilai yang hilang pada kolom 'Score' diisi dengan nilai rata-rata dari kolom tersebut.
- Penanganan data yang hilang diperlukan untuk memastikan bahwa tidak ada nilai yang kosong atau tidak valid dalam data. Hal ini diperlukan karena sebagian besar algoritma machine learning tidak dapat menghandle data yang hilang. Dalam kasus ini, nilai yang hilang pada kolom 'Score' diisi dengan nilai rata-rata dari kolom tersebut agar tidak mengganggu perhitungan dan analisis.
   
### 5. Normalisasi Fitur
- Fitur-fitur tertentu seperti 'Score', 'Type', 'Rating', dan genre anime dinormalisasi menggunakan teknik MinMaxScaler untuk menormalkan rentang nilai.
- Standarisasi data diperlukan untuk memastikan bahwa fitur-fitur memiliki skala yang seragam. Hal ini penting karena beberapa algoritma machine learning sensitif terhadap skala fitur, sehingga standarisasi memungkinkan algoritma tersebut untuk bekerja secara efisien dan menghasilkan hasil yang lebih baik.
- Data skala yang telah dinormalisasi digunakan sebagai masukan untuk menghitung kemiripan antara anime menggunakan cosine similarity. Data yang sudah dinormalisasi memastikan bahwa setiap fitur memiliki rentang nilai yang seragam, sehingga perhitungan kemiripan dapat dilakukan dengan benar dan akurat.

# Modeling
## Pendekatan
Tahapan ini membahas mengenai model sistem rekomendasi yang dibuat untuk menyelesaikan permasalahan. Sistem ini menggunakan pendekatan Content-Based Filtering untuk merekomendasikan anime kepada pengguna berdasarkan kesamaan fitur.
Pendekatan yang digunakan dalam sistem rekomendasi ini adalah Content-Based Filtering. Hal ini dipilih karena pendekatan ini memanfaatkan informasi tentang fitur-fitur suatu item (anime) untuk menemukan kesamaan antara item yang direkomendasikan dengan item yang telah disukai oleh pengguna. Dalam konteks ini, informasi tentang fitur anime seperti genre, rating, dan tipe (TV, Movie, OVA, dll) digunakan untuk menemukan kesamaan.

## Algoritma/Model
Algoritma yang digunakan adalah cosine similarity. Cosine similarity mengukur kesamaan antara dua vektor non-nol dalam ruang berdimensi banyak, yang dalam hal ini mewakili fitur-fitur anime. Cosine similarity dipilih karena cocok untuk membandingkan kesamaan antara vektor fitur yang jarang, seperti fitur genre anime yang biasanya hanya memiliki beberapa nilai yang relevan.

## Tahapan dan Parameter Pemodelan
- Preprocessing Data: Data anime dimuat menggunakan Pandas dan beberapa fitur yang tidak relevan dihapus.
- Konversi Fitur Kategorikal: Fitur kategorikal seperti 'Genres' dan 'Type' diubah menjadi representasi biner menggunakan one-hot encoding.
- Normalisasi Fitur: Fitur-fitur yang akan digunakan untuk perhitungan cosine similarity dinormalisasi menggunakan MinMaxScaler agar memiliki rentang nilai antara 0 dan 1.
- Perhitungan Similaritas: Cosine similarity dihitung antara setiap pasang anime berdasarkan fitur-fitur yang telah dinormalisasi.
- Rekomendasi: Berdasarkan similarity matrix yang dihasilkan, sistem memberikan rekomendasi top-N anime yang paling mirip dengan anime yang telah disukai oleh pengguna.

### Fitur yang Digunakan sebagai Prediksi
Fitur-fitur ini digunakan memungkinkan sistem untuk menemukan kesamaan antara anime yang direkomendasikan dan anime yang telah disukai oleh pengguna.

Fitur yang digunakan untuk prediksi meliputi:

- **Score**: Skor rating anime yang telah dinormalisasi untuk mencerminkan kualitas anime.
- **Type**: Jenis anime seperti TV, Movie, OVA, dll, yang diubah menjadi representasi biner untuk menunjukkan jenis anime.
- **Rating**: Rating konten anime seperti G - All Ages, PG - Children, dll, yang diubah menjadi representasi numerik.
- **Genre**: Genre-genre anime seperti Action, Adventure, Comedy, dll, yang diubah menjadi representasi biner menggunakan one-hot encoding.
    - **Action**: Kehadiran genre Action dalam anime.
    - **Adventure**: Kehadiran genre Adventure dalam anime.
    - **Cars**: Kehadiran genre Cars dalam anime.
    - **Comedy**: Kehadiran genre Comedy dalam anime.
    - **Dementia**: Kehadiran genre Dementia dalam anime.
    - **Demons**: Kehadiran genre Demons dalam anime.
    - **Drama**: Kehadiran genre Drama dalam anime.
    - **Ecchi**: Kehadiran genre Ecchi dalam anime.
    - **Fantasy**: Kehadiran genre Fantasy dalam anime.
    - **Game**: Kehadiran genre Game dalam anime.
    - **Harem**: Kehadiran genre Harem dalam anime.
    - **Historical**: Kehadiran genre Historical dalam anime.
    - **Horror**: Kehadiran genre Horror dalam anime.
    - **Josei**: Kehadiran genre Josei dalam anime.
    - **Kids**: Kehadiran genre Kids dalam anime.
    - **Magic**: Kehadiran genre Magic dalam anime.
    - **Martial Arts**: Kehadiran genre Martial Arts dalam anime.
    - **Mecha**: Kehadiran genre Mecha dalam anime.
    - **Military**: Kehadiran genre Military dalam anime.
    - **Music**: Kehadiran genre Music dalam anime.
    - **Mystery**: Kehadiran genre Mystery dalam anime.
    - **Parody**: Kehadiran genre Parody dalam anime.
    - **Police**: Kehadiran genre Police dalam anime.
    - **Psychological**: Kehadiran genre Psychological dalam anime.
    - **Romance**: Kehadiran genre Romance dalam anime.
    - **Samurai**: Kehadiran genre Samurai dalam anime.
    - **School**: Kehadiran genre School dalam anime.
    - **Sci-Fi**: Kehadiran genre Sci-Fi dalam anime.
    - **Seinen**: Kehadiran genre Seinen dalam anime.
    - **Shoujo**: Kehadiran genre Shoujo dalam anime.
    - **Shoujo Ai**: Kehadiran genre Shoujo Ai dalam anime.
    - **Shounen**: Kehadiran genre Shounen dalam anime.
    - **Shounen Ai**: Kehadiran genre Shounen Ai dalam anime.
    - **Slice of Life**: Kehadiran genre Slice of Life dalam anime.
    - **Space**: Kehadiran genre Space dalam anime.
    - **Sports**: Kehadiran genre Sports dalam anime.
    - **Super Power**: Kehadiran genre Super Power dalam anime.
    - **Supernatural**: Kehadiran genre Supernatural dalam anime.
    - **Thriller**: Kehadiran genre Thriller dalam anime.
    - **Unknown**: Kehadiran genre Unknown dalam anime.
    - **Vampire**: Kehadiran genre Vampire dalam anime.
    - **Yaoi**: Kehadiran genre Yaoi dalam anime.
    - **Yuri**: Kehadiran genre Yuri dalam anime.
## Tampilan Output
Tabel 1. Hasil Rekomendasi Anime
|   | Nama Anime                     |
|---|-------------------------------|
| 1 | Naruto: Shippuuden            |
| 2 | Boruto: Jump Festa 2016 Special |
| 3 | Dragon Ball Z                 |
| 4 | Dragon Ball Kai               |
| 5 | Dragon Ball Kai (2014)        |

Tabel 1, menampilkan rekomendasi anime untuk nama anime yang input pada variabel `anime_rec = "Naruto"`. Rekomendasi ini didasarkan pada kesamaan fitur anime dengan menggunakan metode consine similarity.

# Evaluation

## Metrik Evaluasi
Dalam proyek ini, metrik evaluasi yang digunakan adalah presisi, recall, dan F1-score. 

Presisi (Precision) mengukur proporsi dari item yang direkomendasikan yang relevan bagi pengguna. Recall mengukur proporsi dari item relevan yang direkomendasikan secara keseluruhan. F1-score adalah rata-rata harmonik dari presisi dan recall, memberikan gambaran yang lebih baik tentang keseimbangan antara keduanya.

## Hasil Proyek
Tabel 2. Evaluasi Matrik Sistem Rekomendasi Anime
Metric          | Score
---------------|-------
F1 Score        | 0.222222
Recall Score   | 0.250000
Precision      | 0.200000

Tabel 2 di atas menampilkan nilai metrik evaluasi untuk sistem rekomendasi anime. Terdapat tiga metrik yang dievaluasi, yaitu F1 Score, Recall Score, dan Precision

Berdasarkan metrik evaluasi yang digunakan, hasil proyek adalah sebagai berikut:
- F1 Score: Nilai F1 Score adalah 0.222222 (22,22%). F1 Score adalah rata-rata harmonik dari presisi dan recall, memberikan gambaran yang lebih baik tentang keseimbangan antara keduanya. Nilai F1 Score menunjukkan seberapa baik sistem dalam memberikan rekomendasi yang relevan dan mencakup anime yang disukai oleh pengguna.
- Recall Score: Nilai Recall Score adalah 0.250000 (25%). Recall Score mengukur kemampuan sistem untuk mengidentifikasi semua item yang relevan. Nilai Recall Score menunjukkan seberapa baik sistem dalam merekomendasikan anime yang telah disukai oleh pengguna.
-Precision: Nilai Precision adalah 0.200000 (20%). Precision mengukur seberapa banyak dari rekomendasi yang diberikan oleh sistem yang relevan dengan preferensi pengguna. Nilai Precision menunjukkan seberapa akurat sistem dalam merekomendasikan anime yang benar-benar disukai oleh pengguna.

## Untuk mendapatkan angka presisi 20%, recall 25%, dan 22,22% F1-score, kita menggunakan rumus-rumus berikut:

- Presisi (Precision):
Presisi dihitung dengan membagi jumlah rekomendasi yang relevan dengan jumlah total rekomendasi yang diberikan oleh sistem. Rumusnya adalah sebagai berikut:
Presisi = (Jumlah rekomendasi yang relevan) / (Jumlah total rekomendasi)
Dalam konteks proyek ini, rekomendasi yang relevan adalah anime yang disukai oleh pengguna dan jumlah total rekomendasi adalah semua anime yang direkomendasikan oleh sistem.

- Recall:
Recall dihitung dengan membagi jumlah rekomendasi yang relevan dengan jumlah total anime yang disukai oleh pengguna. Rumusnya adalah sebagai berikut:
Recall = (Jumlah rekomendasi yang relevan) / (Jumlah total anime yang disukai oleh pengguna)
Dalam konteks proyek ini, rekomendasi yang relevan adalah anime yang disukai oleh pengguna dan jumlah total anime yang disukai oleh pengguna adalah semua anime yang telah mereka pilih sebagai favorit.

- F1-score:
F1-score adalah rata-rata harmonik dari presisi dan recall, yang memberikan gambaran yang lebih baik tentang keseimbangan antara keduanya. Rumusnya adalah sebagai berikut:
F1-score = 2 * ((Presisi * Recall) / (Presisi + Recall))
F1-score mencapai nilai maksimum (1) jika dan hanya jika presisi dan recall keduanya mencapai nilai maksimum (1). 

Dalam kode Python yang disediakan, fungsi calculate_precision() digunakan untuk menghitung presisi, sedangkan fungsi calculate_recall() digunakan untuk menghitung recall. Angka presisi, recall, dan F1-score kemudian dicetak untuk dievaluasi.

# Kesimpulan
Proyek ini telah berhasil mengembangkan sistem rekomendasi anime menggunakan pendekatan Content-Based Filtering. Dengan memanfaatkan fitur-fitur anime seperti skor, jenis, rating, dan genre, sistem ini mampu memberikan rekomendasi yang relevan kepada pengguna. Hasil evaluasi menunjukkan bahwa sistem telah mencapai tingkat kinerja yang dapat diterima dengan presisi 20%, recall 25%, dan 22,22% F1-score yang cukup baik. Oleh karena itu, proyek ini telah mampu menyelesaikan permasalahan yang diangkat pada latar belakang dan mencapai tujuan yang diinginkan.

#Referensi
[1] W. Albarqah, I. Irma, and D. K. Izmayanti, “Anime as Japanese Popular Culture,” Abstract of Undergraduate Research, Faculty of Humanities, Bung Hatta University, vol. 1, no. 3, Feb. 2020, Accessed: Mar. 13, 2024. [Online]. Available: https://ejurnal.bunghatta.ac.id/index.php/JFIB/article/view/16150
‌
