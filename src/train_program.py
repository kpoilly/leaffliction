import os
import sys
import argparse

from train import train, load_split_dataset

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/fmaug\
uin/Documents/leaffliction/cuda_local'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if __name__ == "__main__":
    data_path = "dataset"

    parser = argparse.ArgumentParser(
        description="Train the model with custom parameters.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--nb_filters", type=int, default=48,
                        help="Number of filters in the convolutional layers.")
    parser.add_argument("--dropout", type=float,
                        default=0.3, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Patience for early stopping.")
    args = parser.parse_args()

    def train_model(name, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset path '{dataset_path}' not found")
        df_train, df_val = load_split_dataset(dataset_path, args.batch_size)
        model = train(df_train, df_val, name, args.nb_filters,
                      args.dropout, args.epochs, args.patience)
        model.save(f"model/model_{name}.keras")
        print(f"Model saved at 'model/model_{name}.keras'.")

    try:
        # Train the model with the original augmented dataset
        train_model("original", os.path.join(data_path, "original"))
        # Train the model with the mask dataset
        train_model("mask", os.path.join(data_path, "mask"))

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit()
