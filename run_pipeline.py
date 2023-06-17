from Preporcessor import Preprocessor
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="path to testing images")
parser.add_argument("--model_path", type=str,  help="path to Stored Trained_Model")
args = parser.parse_args()

Pretrained_Model = tf.keras.models.load_model(args.model_path)
pepr_= Preprocessor(Path=args.data_path,Mode="Test")
Test_X,Test_Y = pepr_.fit_transform()
Scores =Pretrained_Model.evaluate(Test_X,Test_Y)

print(f"MODEL LOSS==>> {Scores[0]}")
print(f"MODEL ACCURACY==>> {Scores[1]}")
