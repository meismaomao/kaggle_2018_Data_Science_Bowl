import numpy as np
import cv2
import os
from skimage.measure import label
import matplotlib.pyplot as plt

img = cv2.imread(r'H:\data\00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552\images\00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552.png')
mask = np.zeros((256, 256, 1))
mask_list = os.listdir(r'H:\data\00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552\masks')
for index, f in enumerate(mask_list):
    id = index
    path = os.path.join(r'H:\data\00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552\masks', f)
    img = cv2.imread(path, cv2.COLOR_BGR2GRAY)
    img[img == 255] = id
    img_ = np.expand_dims(img, axis=-1)
    mask = np.maximum(mask, img_)

lab_img = label(mask)
run_lengths = []
prev = -2
for i in range(1, lab_img.max() + 1):
    x = (lab_img == i).astype(np.uint8)
    # print(x)
    x = np.transpose(x)
    dots = np.where(x.flatten() == 1)[0]
    # print(dots)
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b

print(run_lengths)

