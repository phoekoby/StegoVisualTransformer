import cv2
import numpy as np
from glob import glob
files = []
for filename in glob('ForCreationDataset256Images/**/*.jpg', recursive=True):
    files.append(filename)

print(len(files))

first = 40001

for i in range(60_000):
    filename = files[i]
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(f'NewImagesFromNewDataset/cover/{i+first}.pgm', image)


# i = 40001
# print(onlyfiles)
# for file in onlyfiles:
#     print(file)
# image = cv2.imread('Linnaeus 5 256X256/train/dog/10_256.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('sample.pgm', image)

