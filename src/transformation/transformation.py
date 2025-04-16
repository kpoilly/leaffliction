from plantcv import plantcv as pcv
import cv2
import os
from .color_histogram import plot_histogram
from rembg import remove
import matplotlib
matplotlib.use('TkAgg')


def remove_background_rembg(image):
    shadow_mask = pcv.rgb2gray_lab(image, channel='l')
    shadow_mask = pcv.threshold.binary(shadow_mask, 1, 'light')
    shadow_mask = pcv.fill(bin_img=shadow_mask, size=500)
    shadow_mask = pcv.erode(shadow_mask, 5, 1)

    result = remove(image)
    grey_scale = pcv.rgb2gray_lab(result, channel='l')
    mask_withoutbg = pcv.threshold.binary(grey_scale, 20, 'light')
    mask_withoutbg = pcv.logical_and(shadow_mask, mask_withoutbg)
    return pcv.fill_holes(bin_img=mask_withoutbg)


class ImgTransformation:
    def __init__(self, img, dst=None, pcv_option=None):
        self.img = img
        self.dst = dst

        self._filled = remove_background_rembg(self.img)

        self._pcv_option = pcv_option
        self._disease_mask = None
        self._g_blur = None
        self._mask = None
        self._roi = None
        self._kept_mask = None
        self._analyzed = None
        self._pseudolandmarks = None

    def get_images(self):
        self.original()
        if self._disease_mask is None:
            self.mask_disease()
        if self._g_blur is None:
            self.gaussian_blur()
        if self._roi is None:
            self.roi_objects()
        if self._analyzed is None:
            self.analyze_objects()
        if self._pseudolandmarks is None:
            self.pseudolandmarks()

        def convert(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return {
            "Original": convert(self.img),
            "Disease_mask": convert(self._disease_mask),
            "Gaussian blur": convert(self._g_blur),
            "ROI objects": convert(self._roi),
            "Analyze object": convert(self._analyzed),
            "Pseudolandmarks": convert(self._pseudolandmarks),
        }

    def change_filename(self, old_name, new_name):
        if self._pcv_option == "print":
            new_path = "{0}_{1}.JPG".format(self.dst, new_name)
            os.rename(old_name, new_path)

    def _print_image(self, img, filename):
        if self._pcv_option == "print":
            pcv.print_image(
                img=img,
                filename=filename
            )
        elif self._pcv_option == "plot":
            pcv.plot_image(img=img, title=filename,)

    def original(self):
        # Save the original image
        self._print_image(
            img=self.img,
            filename="{0}_original.JPG".format(self.dst)
        )
        return self.img

    def mask_disease(self):
        mask = pcv.threshold.dual_channels(self.img,
                                           x_channel="a",
                                           y_channel="b",
                                           points=[(35, 50), (90, 110)],
                                           above=True
                                           )

        self._disease_mask = pcv.logical_xor(self._filled, mask)
        self._print_image(
            img=self._disease_mask,
            filename="{0}_disease_mask.JPG".format(self.dst)
        )
        return self._disease_mask

    def gaussian_blur(self):
        if self._disease_mask is None:
            self.mask_disease()

        # Apply Gaussian blur to the filled image
        self._g_blur = pcv.gaussian_blur(
            img=self._disease_mask, ksize=(3, 3), sigma_x=0, sigma_y=0
        )
        self._print_image(
            img=self._g_blur,
            filename="{0}_gaussian_blur.JPG".format(self.dst)
        )
        return self._g_blur

    def roi_objects(self):
        if self._disease_mask is None:
            self.mask_disease()

        # Create the ROI
        roi = pcv.roi.rectangle(img=self._disease_mask, x=0, y=0,
                                h=self.img.shape[0],
                                w=self.img.shape[1])

        # Create a mask that we will inverse to get green spots
        self._kept_mask = pcv.roi.filter(
            mask=self._disease_mask, roi=roi, roi_type='partial',
        )
        roi_image = self.img.copy()
        roi_image[self._kept_mask != 0] = (0, 255, 0)

        border_size = 5
        roi_image[:border_size, :] = (255, 0, 0)
        roi_image[-border_size:, :] = (255, 0, 0)
        roi_image[:, :border_size] = (255, 0, 0)
        roi_image[:, -border_size:] = (255, 0, 0)

        self._roi = roi_image

        self._print_image(
            img=self._roi,
            filename="{0}_roi_objects.JPG".format(self.dst)
        )
        # Save the mask
        return self._roi

    def analyze_objects(self):
        if self._kept_mask is None:
            self.roi_objects()

        # Analyze the objects in the mask
        self._analyzed = pcv.analyze.size(
            img=self.img, labeled_mask=self._kept_mask)
        self._print_image(
            img=self._analyzed,
            filename="{0}_analyze_object.JPG".format(self.dst)
        )
        return self._analyzed

    def pseudolandmarks(self):
        if self._kept_mask is None:
            self.roi_objects()

        pcv.params.debug = "print"
        # Create the pseudolandmarks
        pcv.homology.x_axis_pseudolandmarks(
            img=self.img, mask=self._kept_mask, label='default'
        )
        pcv.params.debug = None
        filename = f"{str(pcv.params.device - 1)}_x_axis_pseudolandmarks.png"

        self._pseudolandmarks, _, _ = pcv.readimage(filename)
        if self._pcv_option != "print":
            os.remove(filename)

        self.change_filename(
            filename,
            "pseudolandmarks")

        self._print_image(
            img=self._pseudolandmarks,
            filename="{0}_pseudolandmarks.JPG".format(self.dst)
        )
        return self._pseudolandmarks

    def color_histogram(self, display_func=None):
        if self._kept_mask is None:
            self.roi_objects()

        plot_histogram(self.img, self._kept_mask, display_func)


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
    images = cls.get_images()
    if pcv_option == "plot":
        cls.color_histogram()
    elif pcv_option == "print":
        cv2.imwrite(
            f"{dst}_original.JPG",
            images["Original"]
        )


def transformation_from_img(img, displayFunc=None):
    """_summary_
    Augment images in the given path using OpenCV.
    This function applies various transformations to the images
    such as rotate, flip, skew, sher, crop and distortion.

    Args:
        path (_type_): img file path
        to be augmented
    """
    cls = ImgTransformation(img, None)
    images = cls.get_images()
    cls.color_histogram(displayFunc)
    return images
