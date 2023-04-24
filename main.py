import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import confusion_matrix
import numpy as np

# класи, які будемо використовувати для класифікації
classes = [0, 1, 2]

# завантаження набору даних Fashion MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# вибір підмножини даних для трьох класів
train_mask = np.isin(y_train, classes)
test_mask = np.isin(y_test, classes)
X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

# нормалізація значень пікселів до діапазону [0, 1]
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# створення моделі нейронної мережі
model = keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# компіляція моделі
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# навчання моделі
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# використання моделі для класифікації тестового набору даних
y_pred = np.argmax(model.predict(X_test), axis=-1)
# print(y_pred)
# обчислення матриці неточностей (confusion matrix)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
