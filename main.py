import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from skimage.measure import label, regionprops
import numpy as np
import math


def extract_features(image):
    features = []
    # [NEXT, PREVIOUS, FIRST_CHILD, PARENT]
    _, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    ext_cnt = 0
    int_cnt = 0
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][-1] == -1:
            ext_cnt += 1
        elif hierarchy[0][i][-1] == 0:
            int_cnt += 1
    features.extend([ext_cnt, int_cnt])
    
    labeled = label(image)
    region = regionprops(labeled)[0]
    filling_factor = region.area / region.bbox_area
    features.append(filling_factor)

    centroid = np.array(region.centroid) / np.array(region.image.shape)
    features.extend(centroid)
    features.append(region.eccentricity)
    return features


def extract_symbols(image):
    symbols = []

    distances = []
    regions = regionprops(label(image))
    for region in regions:
        _, minx, _, _ = region.bbox
        distances.append([minx, region])
    # a sorting function for array of pairs (arrays), sort by minimum x of a region
    cmp = lambda x: (x[0])
    distances = sorted(distances, key=cmp)

    # this will connect two pieces of i between each other if they are in i_match distance on X axis
    i_match = 12
    i_found = False
    for i in range(len(distances) - 1):
        # print(distances[i][0])
        if i_found:
            i_found = False
            continue
        if distances[i + 1][0] - distances[i][0] > i_match:
            # symbols.append(distances[i][1].image.astype(int))
            # It's preferable to not use region.image.astype(int) because it's somehow ruining the features
            miny, minx, maxy, maxx = distances[i][1].bbox 
            symbols.append(image[miny:maxy, minx:maxx])
            continue
        # if two pieces found, they'll be connected by min-maxing regions
        i_found = True
        miny1, minx1, maxy1, maxx1 = distances[i][1].bbox
        miny2, minx2, maxy2, maxx2 = distances[i + 1][1].bbox
        miny = min(miny1, miny2)
        minx = min(minx1, minx2)
        maxy = max(maxy1, maxy2)
        maxx = max(maxx1, maxx2)
        # symbols.append(image[miny:maxy, minx:maxx].astype(int))
        symbols.append(image[miny:maxy, minx:maxx])
    if not i_found:
        miny, minx, maxy, maxx = distances[-1][1].bbox 
        symbols.append(image[miny:maxy, minx:maxx])
        # symbols.append(distances[-1][1].image.astype(int))
    return symbols


def image_to_text(image, knn):
    text = ""
    symbols = extract_symbols(image)
    for symbol in symbols:
        plt.imshow(symbol)
        plt.show()
        fsymbol = extract_features(symbol)
        fsymbol = np.array(fsymbol, dtype="f4").reshape(1, 6)
        ret, _, _, _ = knn.findNearest(fsymbol, 3)
        text += chr(int(ret))
    return text


train_dir = Path("out") / "train"
train_data = defaultdict(list)

for path in sorted(train_dir.glob("*")):
    if path.is_dir():
        for img_path in path.glob("*.png"):
            symbol = path.name[-1]
            gray = cv2.imread(str(img_path), 0)
            binary = gray.copy()
            binary[binary > 0] = 1
            train_data[symbol].append(binary)

features_array = []
responses = []
for i, symbol in enumerate(train_data):
    print(i)
    for img in train_data[symbol]:
        features = extract_features(img)
        features_array.append(features)
        responses.append(ord(symbol))
features_array = np.array(features_array, dtype="f4")
responses = np.array(responses)

knn = cv2.ml.KNearest_create()
knn.train(features_array, cv2.ml.ROW_SAMPLE, responses)

test_dir = Path("out")
for img_path in sorted(test_dir.glob("*")):
    if img_path.is_dir():
        continue
    print(img_path)
    gray = cv2.imread(str(img_path), 0)
    binary = gray.copy()
    binary[binary > 0] = 1
    print(image_to_text(binary, knn))
