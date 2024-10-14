from glob import glob
from sklearn.cluster import DBSCAN

import numpy as np

# filelist = glob("/Volumes/T7/AIC/data/embeddings/Videos_L01/L01_V001/*.npy")
filelist = glob("/Volumes/T7/AIC/data/embeddings/Videos_L22/L22_V001/*.npy")
data = []
for file in filelist:
    data.append(np.load(file))

data = np.stack(data)
print(data.shape)

clustering = DBSCAN(eps=0.1, min_samples=1, metric="cosine").fit(data)
print(np.unique(clustering.labels_, return_counts=True))
