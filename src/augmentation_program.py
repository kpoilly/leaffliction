from utils import Argument, StaticValidators
from augmentation import augmentation, save_images
import sys


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
    cls.add_validator(StaticValidators.validate_imgFile, args.file)
    cls.validate()
    return args


if __name__ == "__main__":
    try:
        args = arguments_logic()
        images = augmentation(args.file)
        save_images(images)
        exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)
