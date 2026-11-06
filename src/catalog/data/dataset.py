from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, paths, images_dir, transform):
        self.paths = list(paths)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.images_dir / self.paths[i]) as im:
            image = self.transform(im.convert("RGB"))
        return image if self.targets is None else (image, self.targets[i])