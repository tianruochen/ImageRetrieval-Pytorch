import os
import pdb

import torch
import torch.utils.data as data

from modules.datasets.data_helpers import default_loader, imresize


class ImagesFromPathList(data.Dataset):
    """A generic data loader that loads images from a list 
        (Based on ImageFolder from pytorch)
    Args:
        root (string): Root directory path.
        images (list): Relative image paths as strings.
        img_resize (int, Default: None): Defines the maximum size of longer image side
        bbxs (list): List of (x1,y1,x2,y2) tuples to crop the query images
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        images_fn (list): List of full image filename
    """

    def __init__(self, path_list, img_resize=None, bbxs=None, transform=None, loader=default_loader):

        if len(path_list) == 0:
            raise (RuntimeError("Dataset contains 0 images!"))

        self.path_list = path_list
        self.img_resize = img_resize
        self.bbxs = bbxs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            image (PIL): Loaded image
        """
        try:
            path = self.path_list[index]
            img = self.loader(path)
            imfullsize = max(img.size)

            if self.bbxs is not None:
                img = img.crop(self.bbxs[index])

            if self.img_resize is not None:
                if self.bbxs is not None:
                    img = imresize(img, self.img_resize * max(img.size) / imfullsize)
                else:
                    img = imresize(img, self.img_resize)

            if self.transform is not None:
                img = self.transform(img)
            return img, path
        except:
            print(path)
            return self.__getitem__(index+1)

    def __len__(self):
        return len(self.path_list)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImagesFromDataList(data.Dataset):
    """A generic data loader that loads images given as an array of pytorch tensors
        (Based on ImageFolder from pytorch)
    Args:
        images (list): Images as tensors.
        transform (callable, optional): A function/transform that image as a tensors
            and returns a transformed version. E.g, ``normalize`` with mean and std
    """

    def __init__(self, images, transform=None):

        if len(images) == 0:
            raise (RuntimeError("Dataset contains 0 images!"))

        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            image (Tensor): Loaded image
        """
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        if len(img.size()):
            img = img.unsqueeze(0)

        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
