import tensorflow  as tf
import tensorflow.keras as keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images /255.0
test_images = test_images / 255.0

train_images = tf.reshape(train_images, (60000, 28,28,1))
test_images = tf.reshape(test_images, (10000, 28,28,1))

model = keras.models.Sequential()

model.add(tf.keras.Input(shape = (28,28,1)))
model.add(keras.layers.Conv2D(64,3, activation = 'relu'))
model.add(keras.layers.Conv2D(64,3,activation = 'relu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(128, 3, activation = 'relu'))
model.add(keras.layers.Conv2D(128, 3, activation = 'relu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax' ))
model.summary()

model.compile( optimizer = 'adam',
              loss = tf.keras.losses.saprse_categorical_crossentropy(),
              metric = ['Accuracy']
)

history_data = model.fit(train_images, train_labels,
                          validation_data  = (test_images, test_labels),
                            batch = 512, epoch  =100)

