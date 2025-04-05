from utils import Argument, StaticValidators
import sys


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


if __name__ == "__main__":
    try:
        args = arguments_logic()
        print(args)
        exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)
