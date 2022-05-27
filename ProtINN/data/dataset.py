import torch
import os

from typing import List
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision.transforms import transforms as T
from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, Optional

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
class ImageFolderPaths(ImageFolder):
    """inherit ImageFolder, but with addition of returning image paths and only selecting the given class folders"""
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_codes: Optional[List[str]] = []
    ):
        self.class_codes = class_codes
        super().__init__(
            root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        # IMG_EXTENSIONS if is_valid_file is None else None, weggehaald uit init
        self.imgs = self.samples

    def _find_classes(self, dir):
        classes = self.class_codes
        class_to_idx = {classes[i]: i for i in  range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        og_tuple = super(ImageFolderPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (og_tuple + (path,))
        return tuple_with_path

class UnNormalize:
    def __init__(self, mus, sigmas):
        self.m = mus
        self.s = sigmas

    def __call__(self, x):
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] * self.s[i] + self.m[i]
        return x

class ImagenetSegments():

    def __init__(self, root_folder, Vars, batch_size):
        super().__init__()

        self.root_folder = root_folder

        self.batch_size = batch_size

        self.n_classes = 1000
        self.img_crop_size = (224, 224)

        self._mu_img = [0.485, 0.456, 0.406]
        self._std_img = [0.229, 0.224, 0.225]

        self._all_one_hot_encodings = torch.eye(self.n_classes)

        # Returns img, class, and filepath
        self.train_data = ImageFolderPaths(self.root_folder, transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(self.img_crop_size),
            T.ToTensor(),
            T.Normalize(self._mu_img, self._std_img),
        ]), target_transform=T.Compose([T.Lambda(self._class_to_one_hot)]), class_codes=Vars.class_codes)

        self.data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=False, sampler=None)

        self.unnormalize_im = UnNormalize(self._mu_img, self._std_img)

    def set_model(self, model):
        self.data_loader.set_model(model)

    def _uniform_noise(self, x):
        return torch.clamp(x + torch.rand_like(x) / 255.,  min=0., max=1.)

    def _class_to_soft_hot(self, y):
        hard_one_hot = self._all_one_hot_encodings[y]

        return hard_one_hot * (1 - 0.05) + 0.05 / self.n_classes

    def _class_to_one_hot(self, y):
        hard_one_hot = self._all_one_hot_encodings[y]

        return hard_one_hot



class ImagenetTarget():

    def __init__(self, root_folder, Vars, batch_size=50):
        super().__init__()

        self.root_folder = root_folder

        self.batch_size = batch_size

        self.img_crop_size = (224, 224)

        self._mu_img = [0.485, 0.456, 0.406]
        self._std_img = [0.229, 0.224, 0.225]

        self.transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(self.img_crop_size),
            T.ToTensor(),
            T.Normalize(self._mu_img, self._std_img),
        ])
        self.unnormalize_im = UnNormalize(self._mu_img, self._std_img)

        self.classes_to_use = Vars.class_codes

        self.data_paths = []
        for cls in self.classes_to_use:
            class_path = os.path.join(self.root_folder,cls)
            files = os.listdir(class_path)
            full_paths = [os.path.join(class_path, f) for f in files]
            self.data_paths.extend(full_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.data_paths)



class ImagenetSurrogate(Dataset):
    def __init__(self, data, target_dict, transform=None):
        self.data = Variable(torch.FloatTensor(data['similarity'])) # this contains the similarity scores between the prototypes and the closest input segment of input image
        
        self.targetpaths = data['image_paths']
        self.target_dict = target_dict

        self.transform = transform
        self.paths = data['segment_paths'] # for visualisations later

    def __getitem__(self, index):
        x = self.data[index]
        t = '/'.join(self.targetpaths[index].split('/')[-2:])
        classless_t = self.targetpaths[index].split('/')[-1]
        p = self.paths[index]

        if self.transform:
            x = self.transform(x)

        x = (x * 10**2).round() / (10**2)

        y = self.target_dict[classless_t]
        y = (y * 10**2).round() / (10**2)

        return x, y, p, t

    def __len__(self):
        return len(self.data)



class ImagenetSurrogateLabels(Dataset):
    def __init__(self, data, transform=None):
        self.data = Variable(torch.FloatTensor(data['similarity'])) # this contains the similarity scores between the prototypes and the closest input segment of input image
        
        self.targetpaths = data['image_paths']

        self.transform = transform
        self.paths = data['segment_paths'] # for visualisations later

    def __getitem__(self, index):
        x = self.data[index]
        t = '/'.join(self.targetpaths[index].split('/')[-2:])
        p = self.paths[index]

        if self.transform:
            x = self.transform(x)

        x = (x * 10**2).round() / (10**2)

        y = t

        return x, y, p, t

    def __len__(self):
        return len(self.data)


class SegmentData():
    def __init__(self, img_list, path_list):
        self.data = img_list
        self.paths = path_list
        
        self.img_crop_size = (224, 224)

        self._mu_img = [0.485, 0.456, 0.406]
        self._std_img = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.img_crop_size),
            T.ToTensor(),
            T.Normalize(self._mu_img, self._std_img),
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index].convert("RGB")
        X = self.transform(img)
        P = self.paths[index]
        return X,0,P
