
import json
from PIL import Image

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def get_centroids(image_folder, annotated_file):

    ann_json = {}
    with open(annotated_file) as f:
        ann_json = json.load(f)

    annotated = ann_json['_via_img_metadata']

    #centroids = [] # {"class": name_of_class, "centroid": (x, y)}
    centroids = []
    for key, values in annotated.items():
        filename = values["filename"]
        filepath = image_folder + filename
        #
        # Get height and width of image
        im = Image.open(filepath)
        im_width, im_height = im.size
        im.close()

	    # Loop through all the regions, calculate the centroid and then scale it
        # Each region has the top-left (x, y) coordinates, plus the height and width of bounding box
        for region in values["regions"]:
            #class_name = region["region_attributes"]["Class"]
            x = region["shape_attributes"]["x"]
            y = region["shape_attributes"]["y"]
            width = region["shape_attributes"]["width"]
            height = region["shape_attributes"]["height"]
            #print("Filename=", filename, ", class=", class_name, ", x=", x, ", y=", y, ", width=", width, ", height=", height)
            centroid_x = x + width // 2
            centroid_y = y + height // 2
            scaled_x = centroid_x / im_width
            scaled_y = centroid_y / im_height
            #print("Centroid_x=", centroid_x, ", centroid_y=", centroid_y, ", scaled_x=", scaled_x, ", scaled_y=", scaled_y)
            #datapoint = {"class": class_name, "centroid": (scaled_x, scaled_y)}
            datapoint = [scaled_x, scaled_y]
            centroids.append(datapoint)

    return centroids


def kmeans_elbow(image_folder, annotated_file):
    centroids = get_centroids(image_folder, annotated_file)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(centroids)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    return centroids


def kmeans_clusters(n_clusters, X):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    x = [c[0] for c in X]
    y = [c[1] for c in X]
    plt.scatter(x, y)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()



# # Script
# image_folder = "/home/forest/Desktop/eva5/ass12_images/"
# ann_file = "/home/forest/Desktop/eva5/smita_via_project_17Oct2020_20h20m.json"
# centroids = kmeans_elbow(image_folder, ann_file)
# kmeans_clusters(5, centroids)



