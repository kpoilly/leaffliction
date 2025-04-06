
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def plot_stat_hist(label, sc=1):
    """
    Retrieve the histogram x and y values and plot them
    """
    y = pcv.outputs.observations['default_1'][f"{label}_frequencies"]['value']
    x = [
        i * sc
        for i in pcv.outputs.observations['default_1'][
            f"{label}_frequencies"
        ]['label']
    ]
    if label == "hue":
        x = x[:int(255 / 2)]
        y = y[:int(255 / 2)]
    if (
        label == "blue-yellow" or
        label == "green-magenta"
    ):
        x = [x + 128 for x in x]
    plt.plot(x, y, label=label)


def plot_histogram(image, kept_mask, display_func=None):
    """
    Plot the histogram of the image
    """

    dict_label = {
        "blue": 1,
        "green": 1,
        "green-magenta": 1,
        "lightness": 2.55,
        "red": 1,
        "blue-yellow": 1,
        "hue": 1,
        "saturation": 2.55,
        "value": 2.55
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
