#!/usr/bin/env python3

import imageio
import numpy as np
from sklearn.cluster import MiniBatchKMeans, MeanShift
import matplotlib.pyplot as plt
import defopt


def main(image_filename, coordinates_filename, *, detection_filename=None,
         bandwidth=60, min_bin_freq=300):
    """Find micro-structures in an input image.

    :param str image_filename: image file containing micro-structures
    :param str coordinates_filename: output .csv file for coordinates
    :param str detection_filename: ouput image for detected micro-structures
    :param float bandwidth: bandwidth parameter of mean-shift algorithm
    :param int min_bin_freq: min. number of points in bins to seed mean-shift
    """
    # load data
    img = imageio.imread(image_filename)
    img_flat = img.reshape(-1, 3)

    # segment pixels using 3 classes
    print('Segmenting image...', end='', flush=True)
    kmeans = MiniBatchKMeans(3).fit(img_flat)
    labels_flat = kmeans.predict(img_flat)
    labels = labels_flat.reshape(img.shape[:2])
    print(' DONE')

    # isolate pixels of micro-structures (darker)
    idx = np.argmax(kmeans.cluster_centers_.mean(-1))
    coordinates = np.stack(np.where(labels == idx), 1)

    # cluster micro-structures pixels
    print('Clustering micro-structures pixels...', end='', flush=True)
    mshift = MeanShift(bandwidth, n_jobs=-1, bin_seeding=True,
                       min_bin_freq=min_bin_freq)
    mshift.fit(coordinates)
    print(' DONE')

    # save coordinates as .csv file
    np.savetxt(coordinates_filename, mshift.cluster_centers_, header='X,Y')

    # save image of detected clusters
    if detection_filename:
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.scatter(mshift.cluster_centers_[:, 1],
                   mshift.cluster_centers_[:, 0], c='r', s=1)
        ax.set_title('{} objets trouvés'.format(len(mshift.cluster_centers_)))
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(detection_filename, dpi=300)


if __name__ == "__main__":
    defopt.run(main)
