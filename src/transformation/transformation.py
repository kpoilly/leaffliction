from plantcv import plantcv as pcv
import cv2
import os


class ImgTransformation:
    def __init__(self, img, dst, pcv_option="plot"):
        self.img = img
        self.dst = dst
        # Remove the background of the image
        # image_without_bg = rembg.remove(img)
        # pcv.plot_image(img=image_without_bg)
        # Convert the image to grayscale
        s = pcv.rgb2gray_hsv(rgb_img=self.img, channel="s")
        # Create a binary image with a threshold
        l_thresh = pcv.threshold.binary(
            gray_img=s, threshold=70, object_type='light'
        )
        # Remove small objects from the binary image that are smaller than 200 pxls
        self._filled = pcv.fill(bin_img=l_thresh, size=200)

        self._pcv_option = pcv_option
        self._roi = None
        self._mask = None
        self._kept_mask = None
        self._analyzed = None
        self._g_blur = None

    def change_filename(self, old_name, new_name):
        if self._pcv_option == "print":
            new_path = "{0}_{1}.JPG".format(self.dst, new_name)
            os.rename(old_name, new_path)

    def original(self):
        # Save the original image
        if self._pcv_option == "print":
            pcv.print_image(
                img=self.img, filename="{0}_original.JPG".format(self.dst))
        elif self._pcv_option == "plot":
            pcv.plot_image(img=self.img)

    def gaussian_blur(self):
        pcv.params.debug = self._pcv_option

        # Apply Gaussian blur to the filled image
        self._g_blur = pcv.gaussian_blur(
            img=self._filled, ksize=(3, 3), sigma_x=0, sigma_y=0
        )
        pcv.params.debug = None
        self.change_filename(
            f"{str(pcv.params.device - 1)}_gaussian_blur.png", "gaussian_blur")

    def mask(self):
        if self._g_blur is None:
            self.gaussian_blur()

        pcv.params.debug = self._pcv_option

        # Create a mask from the filled image
        self._mask = pcv.apply_mask(
            img=self.img, mask=self._g_blur, mask_color='white')

        pcv.params.debug = None
        self.change_filename(
            f"{str(pcv.params.device - 1)}_masked.png", "mask")

    def roi_objects(self):
        if self._mask is None:
            self.mask()

        # Create the roi
        roi = pcv.roi.rectangle(img=self._mask, x=0, y=0,
                                h=self.img.shape[0],
                                w=self.img.shape[1])

        # Create a mask that we will inverse to get green spots
        self._kept_mask = pcv.roi.filter(
            mask=self._filled, roi=roi, roi_type='partial',
        )
        roi_image = self.img.copy()
        roi_image[self._kept_mask != 0] = (0, 255, 0)

        # Draw the roi on the image
        pcv.params.line_color = (255, 0, 0)
        pcv.params.debug = self._pcv_option
        self._roi = pcv.roi.rectangle(
            img=roi_image, x=0, y=0,
            h=self.img.shape[0],
            w=self.img.shape[1]
        )
        pcv.params.line_color = None
        pcv.params.debug = None
        self.change_filename(
            f"{str(pcv.params.device - 1)}_roi.png", "roi_objects")

    def analyze_objects(self):
        if self._kept_mask is None:
            self.roi_objects()

        pcv.params.debug = self._pcv_option

        # Analyze the objects in the mask
        pcv.analyze.size(
            img=self.img, labeled_mask=self._kept_mask)
        pcv.params.debug = None
        self.change_filename(
            f"{str(pcv.params.device - 1)}_shapes.png", "analyze_object")

    def pseudolandmarks(self):
        if self._kept_mask is None:
            self.roi_objects()

        pcv.params.debug = self._pcv_option

        # Create the pseudolandmarks
        pcv.homology.x_axis_pseudolandmarks(
            img=self.img, mask=self._kept_mask, label='default'
        )
        pcv.params.debug = None
        self.change_filename(
            f"{str(pcv.params.device - 1)}_x_axis_pseudolandmarks.png",
            "pseudolandmarks")

    def color_histogram(self):
        if self._kept_mask is None:
            self.roi_objects()

        pcv.params.debug = self._pcv_option
        pcv.analyze.color(
            self.img,
            self._kept_mask,
            colorspaces="all",
            label="default",
        )

        pcv.params.debug = None


def save_images(images, outdir=None):
    """
    Save the images to the given path with the given name.
    """
    for path, image in images:
        if outdir is not None:
            path = os.path.join(outdir, os.path.basename(path))
        cv2.imwrite(path, image)
        print(f"Saved images to {path}")


def transformation(path, dst, pcv_option="plot"):
    """_summary_
    Augment images in the given path using OpenCV.
    This function applies various transformations to the images
    such as rotate, flip, skew, sher, crop and distortion.

    Args:
        path (_type_): img file path
        to be augmented
    """
    img, _, _ = pcv.readimage(filename=path)
    cls = ImgTransformation(img, dst, pcv_option)
    cls.original()
    cls.gaussian_blur()
    cls.mask()
    cls.roi_objects()
    cls.analyze_objects()
    cls.pseudolandmarks()
    cls.color_histogram()
    # cls.color_histogram()
    # images = cls.get_all_img()
    # images = [(
    #     "{0}_{1}.JPG".format(path if save_in_local_folder
    #                          else os.path.basename(path)[:-4],
    #                          img[0]),
    #     img[1]) for img in images]
    # return images


def transformation_from_img(img):
    """_summary_
    Augment images in the given path using OpenCV.
    This function applies various transformations to the images
    such as rotate, flip, skew, sher, crop and distortion.

    Args:
        path (_type_): img file path
        to be augmented
    """
    cls = ImgTransformation(img)
    images = cls.get_all_img()
    return images
