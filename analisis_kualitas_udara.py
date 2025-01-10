import numpy as np # untuk mengolah data
import pandas as pd # untuk mengolah data 
import matplotlib.pyplot as plt # untuk visualisai diagram

from sklearn.preprocessing import LabelEncoder # untuk encode kategori
from sklearn.preprocessing import MinMaxScaler # untuk normalisasi data

import tensorflow as tf # untuk membuat model neural network

from sklearn.neighbors import KNeighborsClassifier # untuk membuat model KNN
from sklearn.metrics import accuracy_score # untuk mengukur akurasi model KNN

data = pd.read_csv('dataset/updated_pollution_dataset.csv')

print(data.head())
print(f'Panjang data: {len(data)}')

plt.figure(figsize=(15,5))

for i,col in enumerate(data):
    plt.subplot(2,5,i+1)
    plt.title(col,fontsize=10)
    plt.hist(data[col])

plt.tight_layout(pad=0.5)
plt.show()

for i in data[:-1]:
    if i!='Air Quality':
        print(i)
        print(f'max: {np.max(data[i])}')
        print(f'min : {np.min(data[i])}')
        print(f'mean: {np.mean(data[i])}')
        print(f'std : {np.std(data[i])}\n')

le = LabelEncoder()
data['Air Quality'] = le.fit_transform(data['Air Quality'])
print(data['Air Quality'])

minmax = MinMaxScaler()
normalized_X = pd.DataFrame(minmax.fit_transform(data.iloc[:, :-1].values))
labels = data.iloc[:, -1].values
print(normalized_X.head())

X_train = normalized_X[:4500]
y_train = labels[:4500]

X_test = normalized_X[4500:]
y_test = labels[4500:]

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accKNN = accuracy_score(y_test, y_pred)
print("Akurasi KNN:", accKNN)

inputs = tf.keras.layers.Input(shape=(9,))
x = tf.keras.layers.Dense(20, activation='relu')(inputs)
x = tf.keras.layers.Dense(20, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.fit(X_train,y_train,batch_size=32,epochs=10,verbose=2)
loss, accNN = model.evaluate(X_test,y_test,batch_size=32,verbose=2)
print("Akurasi NN:", accNN)