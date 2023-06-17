import tensorflow as tf
import pathlib
class Preprocessor:

    def __init__(self,Path, batch_size=64, img_height=299, img_width=299, validation_rate=0.15, Mode="Train",):
        """
        :param Path: Directory where your images are located
        :param batch_size: 
        :param img_height: 
        :param img_width: 
        :param validation_rate: 
        :param Mode: if you want to train model and preporcess your train and validation data Initialize in Train Mode if you wont to prepare data for Test, so in test mode
        """
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.validation_rate = validation_rate
        self.Mode = Mode
        self.Path = pathlib.Path(Path)

    def fit_transform(self):
        """
        funtion return resized and labeled dataset form images Directory
        if function call is in Training mode function return training and validation dataset and if in test mode return data for testing
        :return:
        """
        if self.Mode == "Train":
            train_data_set = tf.keras.utils.image_dataset_from_directory(
                self.Path,
                validation_split=self.validation_rate,
                subset="training",
                seed=111,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                shuffle=True)
            validation_data_set = tf.keras.utils.image_dataset_from_directory(
                self.Path,
                validation_split=self.validation_rate,
                subset="validation",
                seed=111,
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                shuffle=True)
            return train_data_set,validation_data_set
        else:
            Test_data_set = tf.keras.utils.image_dataset_from_directory(
                self.Path,
                image_size=(self.img_height, self.img_width),
                batch_size=320,
                shuffle=False)
            x=list(Test_data_set)[0][0]
            y=list(Test_data_set)[0][1]
            return x,y
