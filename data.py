from os import listdir
from os.path import join
from torch.utils.data import Dataset

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def to_onehot_tensor(class_num, index):
    label = [0 for _ in range(class_num)]
    label[index] = 1
    return torch.Tensor(label)


class LoadDataset(Dataset):
    def __init__(self, config: dict = None):  # lr_size must be valid
        super(LoadDataset, self).__init__()
        self.config = config
        self.num_class = config['class']['num']
        self.images = []
        
        for classname in self.config['class']['name']:
            for filename in listdir(join(self.config['path'], classname)):
                if is_image_file(filename):
                    self.images.append(join(self.config['path'], classname, filename))

        self.transform = Compose([
            Resize(self.config['image_size']),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

  
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transform(image)
        
        class_name = self.images[index].split('/')[-2]
        label = to_onehot_tensor(self.num_class, self.config['class']['name'].index(class_name))

        return image, label


    def __len__(self):
        return len(self.images)