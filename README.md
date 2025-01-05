# Laporan Proyek Machine Learning - Alexander Kent So
## Domain Proyek

Proyek machine learning ini mengangkat isu mengenai **lingkungan**, dengan fokus **klasifikasi kualitas udara**

**Latar Belakang**

![alt text](asset/cover.png)

Polusi udara merupakan ancaman serius bagi ekosistem. Kualitas udara yang buruk tidak hanya memengaruhi kesehatan manusia, tetapi juga merusak lingkungan. Data Indeks Kualitas Udara (AQI) menunjukkan bahwa kualitas udara di Indonesia tergolong buruk, terutama selama musim kemarau. Meskipun demikian, masih banyak orang yang mengabaikan bahaya polusi udara.[[1](https://ayosehat.kemkes.go.id/bahaya-polusi-udara-bagi-kesehatan)] 
Dampak buruk kualitas udara bagi kesehatan meliputi gangguan pada mata, masalah pernapasan, penyakit kardiovaskular, hingga gangguan kognitif.[[1](https://ayosehat.kemkes.go.id/bahaya-polusi-udara-bagi-kesehatan)] 
Melalui proyek ini, diharapkan kesadaran akan pentingnya menjaga kualitas udara yang baik semakin meningkat, sehingga masyarakat Indonesia dapat mengambil langkah yang lebih bijak dalam menangani polusi udara.

## Business Understanding

Dengan adanya model klasifikasi kualitas udara, masyarakat dapat dengan cepat dan bijak mengetahui kondisi udara di sekitar mereka. Hal ini memungkinkan mereka untuk mengambil langkah-langkah yang tepat dalam menghadapi atau mengurangi dampak polusi udara.

### Problem Statements

Berdasarkan latar belakang proyek ini, berikut adalah pernyataan masalah yang ingin diselesaikan:
- pernyataan masalah 1

### Goals

tujuan dari proyek ini adalah:
- Jawaban pernyataan masalah 1

### Solution statements
- Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
- Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Proyek ini menggunakan dataset **Air Quality and Pollution Assessment** oleh **Mujtaba Mateen** yang diambil dari [kaggle](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment) pada tanggal 30 Desember 2024.

|*|**Keterangan**|
| -------- | ------- |
| Jumlah data | 5000 |
| Kondisi data | well-documented & clean |
| Usability | 10.00 |

### Variabel fitur:
- **Temperature (°C)**: Suhu rata-rata di wilayah tersebut.
- **Humidity (%)**: Kelembapan relatif yang tercatat di - wilayah tersebut.
- **PM2.5 Concentration (µg/m³)**: Tingkat partikel halus di udara.
- **PM10 Concentration (µg/m³)**: Tingkat partikel kasar di udara.
- **NO2 Concentration (ppb)**: Tingkat nitrogen dioksida di udara.
- **SO2 Concentration (ppb)**: Tingkat sulfur dioksida di udara.
- **CO Concentration (ppm)**: Tingkat karbon monoksida di udara.
- **Proximity to Industrial Areas (km)**: Jarak ke zona industri terdekat.
- **Population Density (people/km²)**: Jumlah penduduk per kilometer persegi di wilayah tersebut.

### Variabel target:
**Air Quality Levels**: Kualitas udara berdasarkan fitur yang diberikan yang digolongkan dalam 4 kategori, yaitu:
- **Good**(udara bersih dengan tingkat polusi rendah)
- **Moderate**(Kualitas udara yang dapat diterima, namun terdapat beberapa polutan)
- **Poor**(Polusi udara yang terlihat dan dapat menyebabkan masalah kesehatan bagi kelompok sensitif)
- **Hazardous**(Udara sangat tercemar yang menimbulkan risiko kesehatan serius bagi populasi).

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.