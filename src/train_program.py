import sys
from train import train
from utils import load
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    data_path = "data"

    parser = argparse.ArgumentParser(description="Train the model with custom parameters.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training.")
    parser.add_argument("--nb_filters", type=int, default=64, help="Number of filters in the convolutional layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=2, help="Patience for early stopping.")
    args = parser.parse_args()

    try:
        df_train, df_val = load(data_path, 32)
        class_names = df_train.class_names

        with open("model/class_names.txt", "w") as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")

        model = train(df_train, df_val)
        model.save("model/model.keras")
        print('Model saved at "model/model.keras".')
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit()
