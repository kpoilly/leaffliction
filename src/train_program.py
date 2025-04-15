import os
import sys
import argparse

from train import train, load_split_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    data_path = "data"

    parser = argparse.ArgumentParser(description="Train the model with custom parameters.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--nb_filters", type=int, default=48, help="Number of filters in the convolutional layers.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
    args = parser.parse_args()

    try:
        df_train, df_val = load_split_dataset(data_path, 32)
        model = train(df_train, df_val, args.nb_filters, args.dropout, args.epochs, args.patience)
        model.save("model/model.keras")
        print('Model saved at "model/model.keras".')
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit()
