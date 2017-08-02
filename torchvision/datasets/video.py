import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def videoloader(path, frame_num, cliplen,nframe, cropsize, loader, transform):
    average_duration = frame_num/cliplen
    offset = 0
    ret = torch.FloatTensor(cliplen, 3, cropsize, cropsize)
    for i in xrange(cliplen):
        idx = 0
        if average_duration >= nframe:
            idx = random.randint(1,average_duration)
            idx+= i*average_duration
        frame_id = 0
        for j in xrange(idx, idx+nframe):
            frame_path = '{}/image_{:04d}.jpg'.format(path, j)
            frame = loader(frame_path)
            frame = transform(frame)
            ret[i].copy_(frame)
            frame_id+=1
    return ret		

def read_list(filelist):
    with open(filelist, 'r') as f:
 	_list = []
	for line in f.readlines():
            path, frame_num, label = line.strip().split(' ')
            _list.append((path, int(frame_num), int(label)))	
        return _list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Video(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png 100 0
        root/dog/xxy.png 200 1 

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, filelist,cliplen,istrain,cropsize, transform=None, target_transform=None,
                 loader=default_loader):
        self.videos = read_list(filelist)
        self.cliplen = cliplen
        self.istrain = istrain
        self.cropsize = cropsize
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, frame_num, target = self.videos[index]
        img = videoloader(path, frame_num, self.cliplen,1,  self.cropsize, self.loader, self.transform)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.videos)
