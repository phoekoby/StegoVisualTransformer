import cv2
import numpy as np

algorithm = "S-UNIWARD"

img_cover = cv2.imread("./algorithm_visualize/cover.pgm", cv2.IMREAD_GRAYSCALE)

img_stego = cv2.imread(f"./algorithm_visualize/{algorithm}.pgm", cv2.IMREAD_GRAYSCALE)

img_cover = cv2.cvtColor(img_cover, cv2.COLOR_GRAY2BGR)
img_stego = cv2.cvtColor(img_stego, cv2.COLOR_GRAY2BGR)

cv2.imwrite(f"./algorithm_visualize/cover.png", img_cover)
# cv2.imshow("Cover Image", img_cover)
# # cv2.waitKey(0)
#
# cv2.imshow("Stego Image", img_stego)


pixels = []
for i in range(img_stego.shape[0]):
    for j in range(img_stego.shape[1]):
        found = False
        for b in range(img_stego.shape[2]):
            if img_cover[i, j, b] != img_stego[i, j, b]:
                found = True

        if found:
            pixels.append((i, j))

# print(pixels)

result_image = img_cover.copy()
green = [128, 255, 0]

for pixel in pixels:
    result_image[pixel[0], pixel[1]] = green

# cv2.imshow("Result", result_image)

gray = [132, 132, 132]
white = [0, 0, 0]
black = [255, 255, 255]

new_image = np.full(img_cover.shape, 132)
print(img_cover.shape)
print(new_image.shape)


cv2.imwrite('./algorithm_visualize/gray.png', new_image)

gray_image = cv2.imread("./algorithm_visualize/gray.png", cv2.IMREAD_COLOR)

for pixel in pixels:
    if sum(img_cover[pixel[0], pixel[1]]) - sum(img_stego[pixel[0], pixel[1]]) > 0:
        gray_image[pixel[0], pixel[1]] = white
    else:
        gray_image[pixel[0], pixel[1]] = black

cv2.imwrite(f'./algorithm_visualize/{algorithm}_mask.png', gray_image)
cv2.imshow("Result2", gray_image)
cv2.waitKey(0)
