#!/usr/bin/env python3

import imageio
import numpy as np
from sklearn.cluster import MiniBatchKMeans, MeanShift
import matplotlib.pyplot as plt

# load data
img = imageio.imread('micro_echelle.jpg')
img_flat = img.reshape(-1, 3)

# segment pixels using 3 classes
kmeans = MiniBatchKMeans(3).fit(img_flat)
labels_flat = kmeans.predict(img_flat)
labels = labels_flat.reshape(img.shape[:2])

# isolate pixels of micro-structures (darker)
idx = np.argmax(kmeans.cluster_centers_.mean(-1))
coordinates = np.stack(np.where(labels == idx), 1)

# cluster micro-structures pixels
bandwidth = 60  # ad hoc, improvable
min_bin_freq = 300  # ad hoc, improvable
mshift = MeanShift(bandwidth, n_jobs=-1, bin_seeding=True,
                   min_bin_freq=min_bin_freq)
mshift.fit(coordinates)

# save coordinates as .csv file
np.savetxt('coordinates.csv', mshift.cluster_centers_, header='X,Y')

# save image of detected clusters
fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(mshift.cluster_centers_[:, 1],
           mshift.cluster_centers_[:, 0], c='r', s=1)
ax.set_title('{} objets trouvés'.format(len(mshift.cluster_centers_)))
ax.axis('off')
fig.savefig('detection.png', bbox_inches='tight', dpi=300)
