import tensorflow as tf
from Preporcessor import Preprocessor
Prepr_ = Preprocessor()

training_data, validation_data = Prepr_.fit_transform()

Base_model = tf.keras.applications.InceptionV3(include_top=False,weights="imagenet")
Base_model.trainable = False
model = tf.keras.Sequential()
model.add(tf.keras.layers.Rescaling(scale=1./127.5, offset=-1))
model.add(Base_model)
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(16,activation="sigmoid"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.build(input_shape=(64,299,299,3))
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

