import PIL
class DirConverStegoDataset:
    def __init__(self, img_dir, amount, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.amount = amount

    def __len__(self):
        return self.amount

    def __getitem__(self, idx):
        image = None
        num = int(idx / 2) + 1
        label = idx % 2
        if label == 0:
            image = PIL.Image.open(self.img_dir + "cover/" + str(num) + ".pgm")
        else:
            image = PIL.Image.open(self.img_dir + "stego/" + str(num) + ".pgm")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            print(idx)
        idx += 1
        return (image, label)