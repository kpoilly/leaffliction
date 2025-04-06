from utils import Argument, StaticValidators
import sys
import os
from transformation import transformation, save_images


def arguments_logic():
    cls = Argument(
        "Create images with different transformation \
from an image/dir passed as parameters")
    cls.add_argument(
        "-src",
        str,
        "Path to the img file",
    )
    cls.add_argument(
        "-dst",
        str,
        "Path to the directory to save the images",
        default="output/"
    )
    args = cls.get_args()
    cls.add_validator(StaticValidators.validate_path, args.src)
    cls.validate()
    return args


def transformation_dir(path, output_dir):
    """
    This function creates a list of images from a directory.
    """
    allowed_extensions = (".jpg", ".JPG", ".jpeg")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(allowed_extensions):
                img_path = os.path.join(root, file)
                relative_path = os.path.commonpath([path, root])
                new_path = os.path.relpath(root, relative_path)
                output_subdir = os.path.join(output_dir, new_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                output_file_basename = os.path.join(output_dir, file[:-4])

                transformation(img_path, output_file_basename, "print")


if __name__ == "__main__":
    try:
        args = arguments_logic()
        if not os.path.exists(args.dst):
            os.makedirs(args.dst)

        if os.path.isdir(args.src):
            transformation_dir(args.src, args.dst)
        else:
            images = transformation(args.src, args.dst, "plot")
        exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)
