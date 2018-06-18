#!/usr/bin/env python3

import imageio
import numpy as np
from sklearn.cluster import MiniBatchKMeans, MeanShift
import matplotlib.pyplot as plt

# chargement des données
img = imageio.imread('micro_echelle.jpg')
img_flat = img.reshape(-1, 3)

# segmentation de l'histogramme en 3 classes
kmeans = MiniBatchKMeans(3).fit(img_flat)
labels_flat = kmeans.predict(img_flat)
labels = labels_flat.reshape(img.shape[:2])

# coordonnées des pixels correspondant aux micro-structures (plus sombres)
idx = np.argmax(kmeans.cluster_centers_.mean(-1))
coordinates = np.stack(np.where(labels == idx), 1)

# clustering d'un sous-ensemble des données
bandwidth = 60  # par essai/erreur, améliorable
min_bin_freq = 300  # par essai/erreur, améliorable
mshift = MeanShift(bandwidth, n_jobs=-1, bin_seeding=True,
                   min_bin_freq=min_bin_freq)
mshift.fit(coordinates)

# jolie image de résultats et données en .csv
fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(mshift.cluster_centers_[:, 1],
           mshift.cluster_centers_[:, 0], c='r', s=1)
ax.set_title('{} objets trouvés'.format(len(mshift.cluster_centers_)))
ax.axis('off')
fig.savefig('detection.png', bbox_inches='tight', dpi=300)

np.savetxt('coordinates.csv', mshift.cluster_centers_, header='X,Y')
