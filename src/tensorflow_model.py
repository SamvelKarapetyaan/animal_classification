import tensorflow as tf
from Preporcessor import Preprocessor
Prepr_ = Preprocessor(Path="Your train_data_path ",Mode="Train")
## Loading and spltting data into train and validation sets
training_data, validation_data = Prepr_.fit_transform()
### We use GoogleNet/IncpetionV3 as our base model 


Base_model = tf.keras.applications.InceptionV3(include_top=False,weights="imagenet")
Base_model.trainable=False
model = tf.keras.Sequential()
model.add(tf.keras.layers.Rescaling(scale=1./127.5, offset=-1))
model.add(Base_model)
model.add(tf.keras.layers.GlobalAvgPool2D())
## After Inception pre_ptained block we add 3 Danse layers and one Dropout layer and alos chane output layer into 10 layers Danse with softmax
## trainable patamets are ~~58000
model.add(tf.keras.layers.Dense(28,activation="relu"))
model.add(tf.keras.layers.Dense(18,activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(16,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.build(input_shape=(64,299,299,3))
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

print(model.summary())

model.fit(training_data,validation_data=validation_data,epochs=10)
