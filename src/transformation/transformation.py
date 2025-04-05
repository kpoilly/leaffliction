from plantcv import plantcv as pcv
import rembg
import cv2
import os


def draw_pseudolandmarks(img, plms, color, radius):
    for i in range(len(plms)):
        if len(plms[i]) >= 1 and len(plms[i][0]) >= 2:
            center_x = plms[i][0][1]
            center_y = plms[i][0][0]
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <=\
                            radius ** 2:
                        img[x, y] = color
    return img


class ImgTransformation:
    def __init__(self, img):
        self.img = img
        self.roi_object = None
        self._mask = None
        self._mask_analyze = None
        self.blur = None
        self.ab = None
        self.hierarchy = None
        self.kept_mask = None

    def get_all_img(self):
        images = [
            ('gaussian_blur', self.gaussian_blur()),
            ('mask', self.mask()),
            ('roi_objects', self.roi_objects()),
            ('analyze_objects', self.analyze_objects()),
            ('pseudolandmarks', self.pseudolandmarks())]
        return images

    def gaussian_blur(self):
        img = pcv.rgb2gray_hsv(rgb_img=self.img, channel="s")
        s_thresh = pcv.threshold.binary(
            gray_img=img, threshold=60, max_value=255, object_type="light"
        )
        self.blur = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5),
                                      sigma_x=0, sigma_y=None)

        return self.blur

    def mask(self):
        if self.blur is None:
            self.gaussian_blur()

        b = pcv.rgb2gray_lab(rgb_img=self.img, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b, threshold=200, max_value=255, object_type="light"
        )
        bs = pcv.logical_or(bin_img1=self.blur, bin_img2=b_thresh)

        masked = pcv.apply_mask(img=self.img, mask=bs, mask_color="white")

        masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
        masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel="b")

        maskeda_thresh = pcv.threshold.binary(
            gray_img=masked_a, threshold=115, max_value=255,
            object_type="dark"
        )
        maskeda_thresh1 = pcv.threshold.binary(
            gray_img=masked_a, threshold=135, max_value=255,
            object_type="light"
        )
        maskedb_thresh = pcv.threshold.binary(
            gray_img=masked_b, threshold=128, max_value=255,
            object_type="light"
        )

        ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
        ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)

        ab_fill = pcv.fill(bin_img=ab, size=200)

        masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color="white")

        self._mask = masked2
        self.ab = ab_fill
        return masked2

    def roi_objects(self):
        if self._mask is None:
            self.mask()

        id_objects, obj_hierarchy = pcv.find_objects(img=self.img,
                                                     mask=self.ab)

        roi1, roi_hierarchy = pcv.roi.rectangle(img=self.img,
                                                x=0, y=0, h=250,
                                                w=250)

        pcv.params.debug = self.options.debug

        roi_object, hierarchy3, kept_mask, obj_area = pcv.roi_objects(
            img=self.img,
            roi_contour=roi1,
            roi_hierarchy=roi_hierarchy,
            object_contour=id_objects,
            obj_hierarchy=obj_hierarchy,
            roi_type="partial",
        )

        if self.options.debug == "print":
            file_rename = (
                self.options.outdir
                + "/"
                + str(pcv.params.device - 2)
                + "_obj_on_img.png"
            )
            file_delete = (
                self.options.outdir + "/" + str(pcv.params.device - 1)
                + "_roi_mask.png"
            )

            os.remove(file_delete)
            os.rename(file_rename, self.name_save + "_roi_mask.JPG")

        pcv.params.debug = None

        self.roi_object = roi_object
        self.hierarchy = hierarchy3
        self.kept_mask = kept_mask

        return roi_object, hierarchy3, kept_mask, obj_area

    def analyze_objects(self):
        if self.roi_object is None:
            self.roi_objects()

        obj, mask = pcv.object_composition(
            img=self.img, contours=self.roi_object, hierarchy=self.hierarchy
        )

        analysis_image = pcv.analyze_object(
            img=self.img, obj=obj, mask=mask, label="default"
        )

        if self.options.debug == "print":
            pcv.print_image(
                analysis_image,
                filename=self.name_save + "_analysis_objects.JPG",
            )

        self._mask_analyze = mask
        self.obj = obj
        return analysis_image

    def pseudolandmarks(self):
        if self._mask_analyze is None:
            self.analyze_objects()

        pcv.params.debug = "print"

        top_x, bottom_x, center_v_x = pcv.x_axis_pseudolandmarks(
            img=self.img, obj=self.obj, mask=self._mask_analyze, label="default"
        )

        return top_x, bottom_x, center_v_x

    def color_histogram(self):
        if self._mask_analyze is None:
            self.analyze_objects()

        color_histogram = pcv.analyze_color(
            rgb_img=self.img,
            mask=self.kept_mask,
            colorspaces="all",
            label="default",
        )

        pcv.print_image(
            color_histogram,
            filename=self.name_save + "_color_histogram.JPG",
        )
        return color_histogram


def save_images(images, outdir=None):
    """
    Save the images to the given path with the given name.
    """
    for path, image in images:
        if outdir is not None:
            path = os.path.join(outdir, os.path.basename(path))
        cv2.imwrite(path, image)
        print(f"Saved images to {path}")


def transformation(path, save_in_local_folder=False):
    """_summary_
    Augment images in the given path using OpenCV.
    This function applies various transformations to the images
    such as rotate, flip, skew, sher, crop and distortion.

    Args:
        path (_type_): img file path
        to be augmented
    """
    img = pcv.readimage(filename=path)
    cls = ImgTransformation(img)
    cls.analyze_objects()
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
