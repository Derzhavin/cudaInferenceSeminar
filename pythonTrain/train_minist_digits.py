import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential

# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.Adam(epsilon=1e-8)
# optimizer = sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
# optimizer = sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.7, nesterov=True)
# optimizer = sgd = tf.keras.optimizers.SGD(lr=0.05, momentum=0.9, nesterov=True)
# optimizer = sgd = tf.keras.optimizers.SGD(lr=0.05, momentum=0.7, nesterov=True)
# optimizer = tf.keras.optimizers.RMSprop()

mnist = tf.keras.datasets.mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()


plt.imshow(train_images[0],cmap=plt.cm.binary)
plt.show()

print(train_labels[0])

train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20, batch_size=256)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

model.save('mnist_digits_model', save_format="tf")