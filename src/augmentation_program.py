from utils import Argument, StaticValidators
from augmentation import augmentation, save_images
import sys
import os


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


def augmentation_dir(path):
    """
    This function creates a list of images from a directory.
    """
    allowed_extensions = (".jpg", ".JPG", ".jpeg")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(allowed_extensions):
                img_path = os.path.join(root, file)
                images = augmentation(img_path, True)
                save_images(images)


if __name__ == "__main__":
    try:
        args = arguments_logic()
        if os.path.isdir(args.file):
            augmentation_dir(args.file)
        else:
            images = augmentation(args.file)
            save_images(images)
        exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)
