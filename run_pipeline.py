import tensorflow as tf
import argparse

from src.model import Model
from src.preprocessor import Preprocessor


class Pipeline:
    def __init__(self):
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, data):
        # Model and Preprocessor loading process.
        self.model = ... # load network
        self.preprocessor = ... # just load images as dataset

        # Get probabilties and best threshold
        predictions = ... # get predictions for each image
        
        ... # somehow return predictions

def main():
    parser = argparse.ArgumentParser(
        description="""
        It defines one argument that can be passed to the program:

        "--data_path": a required argument that 
            specifies the path to the data file.
        """,    
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add --data_path and --inference arguments to parser
    parser.add_argument("--data_path", help="Path to data file.", required=True)

    # Get arguments as dictionary from parser
    args = parser.parse_args() # returns dictionary-like object

    path_of_data = args.data_path

    # Pipeline running
    pipeline = Pipeline()
    pipeline.run(path_of_data)


if __name__ == "__main__":
    main()