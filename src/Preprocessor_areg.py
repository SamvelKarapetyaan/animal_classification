import tensorflow as tf

class Preprocessor:
    ## can be your custom file path
    path_in_jpeg_format = tf.data.Dataset.list_files(
        r"C:\Users\Dell\Desktop\Animals_project\big_train_data_1008\*\*.jpeg", shuffle=True)
    path_in_jpg_format = tf.data.Dataset.list_files(
        r"C:\Users\Dell\Desktop\Animals_project\big_train_data_1008\*\*.jpg", shuffle=True)
    all_data = path_in_jpeg_format.concatenate(path_in_jpg_format)
    number_of_data_points = len(list(all_data))

    def __init__(self, batch_size=64, img_height=299, img_width=299, validation_rate=0.15):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.validation_rate = validation_rate
        val_size = int(self.number_of_data_points * validation_rate)
        # Splitting data to validaation and training groups
        self.train_ds = self.all_data.skip(val_size)
        self.val_ds = self.all_data.take(val_size)

    @tf.function
    def get_label(self, file_path):
        parts = tf.strings.split(file_path, "\\")
        one_hot = parts[-2] == self.class_names
        return tf.argmax(one_hot)

    @tf.function
    def decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [self.img_height, self.img_width])

    @tf.function
    def process_path(self, file_path):
        label = self.get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    @tf.function
    def configure_for_performance(self, ds):
        ds = ds.cache()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def fit_transform(self):
        lambda_function = lambda x: (tf.strings.split(x, "\\"))[-2]
        self.class_names = [i.numpy() for i in self.all_data.map(lambda_function).unique()]
        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

        train_ds = self.train_ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = self.val_ds.map(self.process_path, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = self.configure_for_performance(train_ds)
        val_ds = self.configure_for_performance(val_ds)

        return (train_ds, val_ds)
