# Laporan Proyek Machine Learning - Alexander Kent So
## Domain Proyek
Proyek machine learning ini mengangkat isu mengenai **lingkungan**, dengan fokus **klasifikasi kualitas udara**. Metode machine learning dapat membantu menyelesaikan masalah ini dengan memberikan klasifikasi yang cepat dan akurat mengenai kualitas udara berdasarkan data yang ada.

**Latar Belakang**

![alt text](asset/cover.png)

Polusi udara merupakan ancaman serius bagi ekosistem. Kualitas udara yang buruk tidak hanya memengaruhi kesehatan manusia, tetapi juga merusak lingkungan. Data Indeks Kualitas Udara (AQI) menunjukkan bahwa kualitas udara di Indonesia tergolong buruk, terutama selama musim kemarau. Meskipun demikian, masih banyak orang yang mengabaikan bahaya polusi udara.[[1](https://ayosehat.kemkes.go.id/bahaya-polusi-udara-bagi-kesehatan)] 
Dampak buruk kualitas udara bagi kesehatan meliputi gangguan pada mata, masalah pernapasan, penyakit kardiovaskular, hingga gangguan kognitif.[[1](https://ayosehat.kemkes.go.id/bahaya-polusi-udara-bagi-kesehatan)] 
Melalui proyek ini, diharapkan kesadaran akan pentingnya menjaga kualitas udara yang baik semakin meningkat, sehingga masyarakat Indonesia dapat mengambil langkah yang lebih bijak dalam menangani polusi udara.

## Business Understanding
Dengan adanya model klasifikasi kualitas udara, masyarakat dapat dengan cepat dan bijak mengetahui kondisi udara di sekitar mereka. Hal ini memungkinkan mereka untuk mengambil langkah-langkah yang tepat dalam menghadapi atau mengurangi dampak polusi udara.

### Problem Statements
Berdasarkan latar belakang proyek ini, berikut adalah pernyataan masalah yang ingin diselesaikan:
- Bagaimana cara menentukan kategori kualitas udara di suatu wilayah berdasarkan data yang ada?
- Apa langkah yang dapat diambil masyarakat untuk mengurangi paparan polusi udara berdasarkan prediksi kualitas udara?

### Goals
tujuan dari proyek ini adalah:
- Membangun model klasifikasi kualitas udara di suatu wilayah berdasarkan data yang tersedia.
- Meningkatkan kesadaran masyarakat tentang pentingnya menjaga kualitas udara dengan menyediakan informasi yang mudah diakses mengenai kualitas udara di sekitar mereka.

### Solution statements
- Proyek ini menggunakan 2 algoritma machine learning, yaitu: KNN dan neural network dari TensorFlow. Neural network dipilih karena fleksibilitasnya dalam mengolah berbagai tipe data. KNN dipilih karena merupakan algoritma klasifikasi yang sederhana. Performa dari 2 algoritma ini akan dibandingkan dengan metrik akurasi untuk menentukan algoritma yang lebih baik.


## Data Understanding
Proyek ini menggunakan dataset **Air Quality and Pollution Assessment** oleh **Mujtaba Mateen** yang diambil dari [kaggle](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment) pada tanggal 30 Desember 2024.

|*|**Keterangan**|
| -------- | ------- |
| Jumlah data | 5000 |
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

![alt text](asset/EDA.png)

| **kolom** | **max** | **min** | **mean** | **std** | **null value** |
| -------- | ------- | -------- | ------- | -------- | ------- |
| Temperature | 58.6 | 13.4 | 30.02902 | 6.719989 | 0 |
| Humidity | 128.1 | 36.0 | 70.05612 | 15.86199 | 0 |
| PM2.5 | 295.0 | 0.0 | 20.14214 | 24.55209 | 0 |
| PM10 | 315.8 | -0.2 | 30.21836 | 27.346464 | 0 |
| NO2  | 64.9 | 7.4 | 26.4121 | 8.894467 | 0 |
| SO2 |44.9 | -6.2 | 10.01482 | 6.749628 | 0 |
| CO | 3.72 | 0.65 | 1.500354 | 0.545972 | 0 |
| Proximity_to_Industrial_Areas | 25.8 | 2.5 | 8.4254 | 3.610583 | 0 |
| Population_Density | 957 | 188 | 497.4238 | 152.738808 | 0 |

```python
duplicate_val = data.duplicated().sum()
print(f'jumlah data duplikat: {duplicate_val}') # output jumlah data duplikat: 0
```

Dapat dilihat pada grafik, potongan code, dan tabel mengenai distribusi data, jumlah data duplikat, dan jumlah nilai null pada setiap kolom. Dapat dilihat juga ada beberapa kolom dengan nilai data yang dapat dikategorikan sebagai outlier, contohnya pada kolom PM10 dan SO2 di mana nilai min mencapai angka minus(-) dan juga pada kolom temperatur dengan nilai max 58.6'C

## Data Preparation
Pada bagian Data Preparation, terdapat 3 metode yang digunakan dalam proyek ini, yaitu:

- **menghapus outlier dengan metode IQR**   : Menghilangkan nilai data outlier dengan membuang nilai data yang ada di luar Interquartile Range(IQR). Pada tahap ini ditambahkan hyperparameter **alpha** yang menentukan range pengambilan data. **alpha** diberi nilai 0.5 dengan pertimbangan agar nilai min dari PM10 dan SO2 tidak bernilai minus(-).

- **Encoding**      : Mengubah data kategori(teks) menjadi angka. Encoding hanya dilakukan pada kolom 'Air Quality' agar komputer dapat memproses data tersebut dengan lebih efisien.

- **Normalisasi**   : Mengubah data numerik ke dalam skala tertentu. Dalam proyek ini, metode Min-Max Normalization digunakan untuk merubah nilai ke rentang [0, 1]. Tujuannya adalah untuk memastikan semua nilai berada dalam skala yang seimbang, sehingga tidak ada fitur yang mendominasi, serta mempermudah pemrosesan data oleh komputer.

- **train-test data split** : Membagi dataset menjadi  2 subset, yaitu: train dataset untuk melatih model dan test dataset untuk menguji performa model terhadap data baru. Pembagiannya adalah 90% data untuk training dan 10% data untuk test.

## Modeling
Sesuai pernyataaan di **solution statement** proyek ini menggunakan 2 algoritma machine learning, yaitu: **KNN(K-Nearest Neighbors)** dan **neural network tensorflow**.

### **KNN(K-Nearest Neighbour)**:
Algoritma yang digunakan untuk klasifikasi data. Pada tahap pelatihan, data akan dipetakan dalam bentuk grafik atau ruang fitur. Untuk mengklasifikasikan data baru, algoritma ini menghitung jarak antara data baru tersebut dengan semua data yang ada dalam training set. Kemudian, k data terdekat yang ditemukan akan dipilih, dan klasifikasi akan dilakukan berdasarkan mayoritas kelas dari k tetangga terdekat tersebut.
- Kelebihan  : Sederhana dan memiliki waktu training yang relatif cepat
- Kekurangan : Waktu prediksi relatif lambat dan boros daya komputasi saat dataset cukup besar
- Hyperparameter : 
**k** yaitu jumlah tetangga terdekat yang diperhitungkan untuk proses klasifikasi data baru, proyek ini menggunakan **k=5** yang didapat dari jumlah kategori(4) + 1 untuk mengantisipasi 2 kategori saling seimbang.
**metric** yaitu metode menghitung jarak antar data, proyek ini menggunakan **metric euclidian** untuk menghitung jarak antar data. Hyperparameter lain akan mengikuti setelan default dari sklearn.

### **Neural Network**:
Algoritma yang terinspirasi oleh cara kerja otak manusia. Pada dasarnya, neural network terdiri dari banyak neuron yang saling terhubung dalam lapisan-lapisan (layer). Proses pelatihan dimulai dengan memberi input pada jaringan, yang kemudian akan diteruskan ke neuron-neuron pada setiap lapisan. Setiap neuron memproses input berdasarkan bobot yang dimilikinya dan menghasilkan output yang diteruskan ke lapisan berikutnya. Pada proses training, bobot tiap neuron akan terus diubah agar dpaat menghasilkan output yang diinginkan. Pada proses klasifikasi, parameter data akan dimasukan ke dalam model dan diproses oleh neuron yang sudah dilatih untuk menentukan klasifikasi dari data tersebut.
- Kelebihan  : Memiliki waktu prediksi yang relatif cepat dan mampu memprediksi data kompleks
- Kekurangan : Waktu training relatif lambat dan boros daya komputasi
- Arsitektur :

![alt text](asset/arsitektur.png)

- Hyperparameter : 
**Activation function** yang digunakan pada layer **Dense** dan **Dense_1** adalah **ReLU** yang umum digunakan pada hidden layer dan **Softmax** pada **Dense_2** yang cocok untuk multiclass classification. 
**Optimizer** yang digunakan adalah **Adam**. 
**Loss function** yang digunakan adalah **sparse_categorical_crossentropy** yang cocok untuk multiclass classification tanpa memerlukan one-hot encoding. 
**Epoch** yang dipakai adalah **Epoch=10** karena dalam 10 iterasi, model sudah mencapai performa yang cukup baik.

## Evaluation
Metric evaluasi yang digunakan pada proyek ini adalah **akurasi**.
Akurasi adalah rasio prediksi yang benar terhadap total prediksi yang dibuat. Formula untuk menghitung akurasi adalah:

$$
\text{Akurasi} = \frac{\text{Jumlah Prediksi Benar}}{\text{Total Prediksi}}
$$

Dimana:
- **Jumlah Prediksi Benar** adalah jumlah contoh yang diklasifikasikan dengan benar oleh model (baik kelas positif maupun negatif).
- **Total Prediksi** adalah jumlah keseluruhan contoh dalam dataset.

Berdasarkan Business Understanding yang sudah dijabarakan sebelumnya, saya menarik kesimpulan bahwa proyek ini mampu menjawab problem statement dan memenuhi goals yang ada.
- Pada proyek ini, model yang dihasilkan mampu mengklasifikasikan kualitas udara berdasarkan data yang ada dengan cepat dan dengan akurasi yang cukup tinggi(>90%).
- Dengan adanya sistem klasifikasi kualitas udara ini, masyarakat memiliki panduan untuk mengklasifikasikan kualitas udara di lingkungan merekadenga cepat dan akurat. Hal yang diharapkan mampu meningkatkan keperdulian masyarakat terhadap kualitas udara di lingkungan mereka.

Solution statement yang dijabarkan juga sudah berdampak dalam proyek ini. Di mana ada 2 model machine learnng yang dibuat, yaitu: KNN dan neural network. Hasil menunjukan bahwa akurasi dari model neural network(~96.66%) mengalahkan akurasi dari model KNN(~94.18%). Sehingga ditarik kesimpulan bahwa dalam proyek ini performa model neural network lebih baik dibanding performa model KNN.

## Referensi
[1] https://ayosehat.kemkes.go.id/bahaya-polusi-udara-bagi-kesehatan
[2] https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment