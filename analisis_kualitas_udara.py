import numpy as np # untuk mengolah data
import pandas as pd # untuk mengolah data 
import matplotlib.pyplot as plt # untuk visualisai diagram
from sklearn.preprocessing import LabelEncoder # untuk encode kategori
from sklearn.preprocessing import MinMaxScaler # untuk normalisasi data
import tensorflow as tf # untuk membuat model neural network
from sklearn.neighbors import KNeighborsClassifier # untuk membuat model KNN
from sklearn.metrics import accuracy_score # untuk mengukur akurasi model KNN

# membaca data
data = pd.read_csv('dataset/updated_pollution_dataset.csv')

# membuang outlier dengan metode IQR
filtered_data = data.copy()
for i in ['Temperature','PM10','SO2']:
    # mencari IQR
    Q1 = filtered_data[i].quantile(0.25)
    Q3 = filtered_data[i].quantile(0.75)
    IQR = Q3 - Q1
    # Batas bawah dan atas
    alpha = 0.50 # mengatur pengambilan data
    lower_bound = Q1 - alpha * IQR
    upper_bound = Q3 + alpha * IQR
    filtered_data = filtered_data[ (filtered_data[i] >= lower_bound) & (filtered_data[i] <= upper_bound) ]

# encode bagian air quality
le = LabelEncoder()
filtered_data['Air Quality'] = le.fit_transform(filtered_data['Air Quality'])

# normalisasi minmax
minmax = MinMaxScaler()
normalized_X = pd.DataFrame(minmax.fit_transform(filtered_data.iloc[:, :-1].values))
labels = filtered_data.iloc[:, -1].values

# split dataset 90%:10%
treshold = int(len(filtered_data)*0.9)
X_train = normalized_X[:treshold]
y_train = labels[:treshold]
X_test = normalized_X[treshold:]
y_test = labels[treshold:]

# KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accKNN = accuracy_score(y_test, y_pred)
print("Akurasi KNN:", accKNN)

# membuat model neural network
inputs = tf.keras.layers.Input(shape=(9,))
x = tf.keras.layers.Dense(20, activation='relu')(inputs)
x = tf.keras.layers.Dense(20, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# klasifikasi neural network
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.fit(X_train,y_train,batch_size=32,epochs=10,verbose=2)
loss, accNN = model.evaluate(X_test,y_test,batch_size=32,verbose=2)
print("Akurasi NN:", accNN)