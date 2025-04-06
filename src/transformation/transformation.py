from plantcv import plantcv as pcv
import cv2
import os
import matplotlib.pyplot as plt


def plot_stat_hist(label, sc=1):
    """
    Retrieve the histogram x and y values and plot them
    """

    y = pcv.outputs.observations['default_1'][label]['value']
    x = [
        i * sc
        for i in pcv.outputs.observations['default_1'][label]['label']
    ]
    if label == "hue_frequencies":
        x = x[:int(255 / 2)]
        y = y[:int(255 / 2)]
    if (
        label == "blue-yellow_frequencies" or
        label == "green-magenta_frequencies"
    ):
        x = [x + 128 for x in x]
    plt.plot(x, y, label=label)


def plot_histogram(image, kept_mask, display_func=None):
    """
    Plot the histogram of the image
    """

    dict_label = {
        "blue_frequencies": 1,
        "green_frequencies": 1,
        "green-magenta_frequencies": 1,
        "lightness_frequencies": 2.55,
        "red_frequencies": 1,
        "blue-yellow_frequencies": 1,
        "hue_frequencies": 1,
        "saturation_frequencies": 2.55,
        "value_frequencies": 2.55
    }

    labels, _ = pcv.create_labels(mask=kept_mask)
    pcv.analyze.color(
        rgb_img=image,
        colorspaces="all",
        labeled_mask=labels,
        label="default"
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    for key, val in dict_label.items():
        plot_stat_hist(key, val)

    plt.legend()

    plt.title("Color Histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")
    plt.grid(
        visible=True,
        which='major',
        axis='both',
        linestyle='--',
    )
    if display_func is not None:
        display_func(fig)
    else:
        plt.show()
        plt.close()


class ImgTransformation:
    def __init__(self, img, dst=None, pcv_option=None):
        self.img = img
        self.dst = dst
        # Convert the image to grayscale
        s = pcv.rgb2gray_hsv(rgb_img=self.img, channel="s")
        # Create a binary image with a threshold
        s_thresh = pcv.threshold.binary(
            gray_img=s, threshold=70, object_type='light'
        )
        # Remove small objects from the binary image that are smaller
        # than 200 pxls
        self._filled = pcv.fill(bin_img=s_thresh, size=200)

        self._pcv_option = pcv_option
        self._g_blur = None
        self._mask = None
        self._roi = None
        self._kept_mask = None
        self._analyzed = None
        self._pseudolandmarks = None

    def get_images(self):
        if self._g_blur is None:
            self.gaussian_blur()
        if self._mask is None:
            self.mask()
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
            "Gaussian blur": convert(self._g_blur),
            "Mask": convert(self._mask),
            "ROI objects": convert(self._roi),
            "Analyze object": convert(self._analyzed),
            "Pseudolandmarks": convert(self._pseudolandmarks),
        }

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

        # Create the ROI
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
        pcv.params.debug = "print"
        self._roi = pcv.roi.rectangle(
            img=roi_image, x=0, y=0,
            h=self.img.shape[0],
            w=self.img.shape[1]
        )
        pcv.params.line_color = None
        pcv.params.debug = None
        filename = f"{str(pcv.params.device - 1)}_roi.png"
        self._roi, _, _ = pcv.readimage(filename)
        if self._pcv_option != "print":
            os.remove(filename)

        self.change_filename(
            filename, "roi_objects")

    def analyze_objects(self):
        if self._kept_mask is None:
            self.roi_objects()

        pcv.params.debug = self._pcv_option

        # Analyze the objects in the mask
        self._analyzed = pcv.analyze.size(
            img=self.img, labeled_mask=self._kept_mask)
        pcv.params.debug = None
        self.change_filename(
            f"{str(pcv.params.device - 1)}_shapes.png", "analyze_object")

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

    def color_histogram(self, display_func=None):
        if self._kept_mask is None:
            self.roi_objects()

        plot_histogram(self.img, self._kept_mask, display_func)


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
