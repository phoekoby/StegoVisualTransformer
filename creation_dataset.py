from PIL import Image
import shutil

# for i in range(1, 10001):
#     img = Image.open('./BOSSbase_1.01/' + str(i) + ".pgm")
#
#     new_img_1 = img.crop((0, 0, 256, 256))
#     new_img_2 = img.crop((256, 256, 512, 512))
#     new_img_3 = img.crop((0, 256, 256, 512))
#     new_img_4 = img.crop((256, 0, 512, 256))
#
#     new_img_1.save('./cropped_256_BOSSbase/' + str((i - 1) * 4 + 1) + '.pgm')
#     new_img_2.save('./cropped_256_BOSSbase/' + str((i - 1) * 4 + 2) + '.pgm')
#     new_img_3.save('./cropped_256_BOSSbase/' + str((i - 1) * 4 + 3) + '.pgm')
#     new_img_4.save('./cropped_256_BOSSbase/' + str((i - 1) * 4 + 4) + '.pgm')


for i in range(1,100_001):
    if i % 3 == 0:
        shutil.copyfile(f'./NewMoreImagesDataset/HUGO-256cropped/stego/{i}.pgm',f'./NewMoreImagesDataset/MULTIAL-256cropped/stego/{i}.pgm')
    if i % 3 == 1:
        shutil.copyfile(f'./NewMoreImagesDataset/WOW-256cropped/stego/{i}.pgm',f'./NewMoreImagesDataset/MULTIAL-256cropped/stego/{i}.pgm')
    if i % 3 == 2:
        shutil.copyfile(f'./NewMoreImagesDataset/S-UNIWARD-256cropped/stego/{i}.pgm',f'./NewMoreImagesDataset/MULTIAL-256cropped/stego/{i}.pgm')