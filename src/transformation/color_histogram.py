
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def plot_stat_hist(label, scl=1):

    # outputs observation of the last pcv
    y = pcv.outputs.observations['default_1'][f"{label}_frequencies"]['value']
    x = [
        i * scl
        for i in pcv.outputs.observations['default_1'][
            f"{label}_frequencies"
        ]['label']
    ]
    # normalize the hue from 0 - 360 to 0 - 180
    if label == "hue":
        x = x[:int(255 / 2)]
        y = y[:int(255 / 2)]

    # recenter the blue-yellow and green-magenta on the 0-255 scale
    # CIELAB a* and b* are unsigned integers
    if (
        label == "blue-yellow" or
        label == "green-magenta"
    ):
        x = [x + 128 for x in x]
    plt.plot(x, y, label=label)


def plot_histogram(image, kept_mask, display_func=None):

    # Create a dictionary to store the labels and their corresponding scale
    # some labels are on % so they are multiplied by 2.55
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

    plt.title("Color histogram")
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
