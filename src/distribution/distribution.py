import os
import matplotlib.pyplot as plt


def distribution(path):
    data_map = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                label = os.path.basename(root)
            if label not in data_map:
                data_map[label] = 0
            data_map[label] += 1

    return data_map


def plot_distribution(data_map):

    labels = list(data_map.keys())
    labels = sorted(labels, key=lambda x: data_map[x], reverse=True)
    values = [data_map[label] for label in labels]

    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab20.colors
    plt.bar(labels, values, color=[colors[i % len(colors)]
            for i in range(len(labels))])
    plt.xlabel('Labels')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images by directory')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=[colors[i % len(colors)] for i in range(len(labels))])
    plt.title('Distribution of Images by directory')
    plt.tight_layout()
    plt.show()
