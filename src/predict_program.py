import os
import sys
from predict import predict
from utils import Argument, StaticValidators


def arguments_logic():
    cls = Argument(
        "Create images with different transformation \
from an image passed as parameters")
    cls.add_argument(
        "file",
        str,
        "Path to the img file",
    )
    args = cls.get_args()
    cls.add_validator(StaticValidators.validate_path, args.file)
    cls.validate()
    return args


if __name__ == "__main__":
    try:
        args = arguments_logic()
        model_path = "model/model.keras"
        image_path = args.file
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Make sure to train the model first!")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image file '{image_path}' not found.")
        
        predicted_class = predict(image_path, model_path)
        print(f"'{image_path}' has been predicted as : {predicted_class}")
        exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)
