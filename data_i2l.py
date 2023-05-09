from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from os.path import join
import torch.nn.functional as F
import numpy as np
import torch

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_image_data(dataset_path, from_zero=True, train_labels=['mug', 'ball']):
    data = {
        'train': {'samples': [], 'labels': []},
        'knowledge': {'samples': [], 'labels': []},
    }
    for ext in IMG_EXTENSIONS:
        for img_path in glob(join(dataset_path, '*', f'*{ext}')):
            label = img_path.split('/')[-2]
            
            data['knowledge']['samples'].append(img_path)
            data['knowledge']['labels'].append(label)

            if not from_zero and label in train_labels:
                data['train']['samples'].append(img_path)
                data['train']['labels'].append(label)

    return data


class I2LDataset(Dataset):

    def __init__(self, x, y, samples, labels):
        self.x = x
        self.y = y
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.samples[index], self.labels[index]

    def __len__(self):
        return self.x.shape[0]
    

class I2LData:

    def __init__(self, net, data):
        self.net = net
        self.data = data

        self.labels = None
        self.target2label = None
        self.pt = None
        self.pk = None
        self.n = None
        self.m = None
        self.x = {'train': None, 'dev': None, 'knowledge': None}
        self.y = {'train': None, 'dev': None, 'knowledge': None}
        self.init_data()

    def init_data(self):
        self.labels = sorted(list(set(self.data['knowledge']['labels'])))
        self.target2label = {target:label for target, label in enumerate(self.labels)}

        self.data['train']['targets'] = [self.labels.index(label) for label in self.data['train']['labels']]
        self.data['knowledge']['targets'] = [self.labels.index(label) for label in self.data['knowledge']['labels']]
        
        self.data['train']['encodings'] = self.encode_samples(self.data['train']['samples'])
        self.data['knowledge']['encodings'] = self.encode_samples(self.data['knowledge']['samples'])

        self.data['dev'] = {
            'samples': self.data['knowledge']['samples'][:],
            'encodings': self.data['knowledge']['encodings'].copy(),
            'labels': self.data['knowledge']['labels'][:],
            'targets': self.data['knowledge']['targets'][:]
        }

        self.pt = len(self.data['train']['samples'])
        self.pk = len(self.data['knowledge']['samples'])

        self.make_pairs(('knowledge', 'dev', 'train'))

        self.n = len(self.x['knowledge'][0])
        self.m = len(self.labels)

        print(f'Required knowledge initialized: n = {self.n}, m = {self.m}, pt = {self.pt}, pk = {self.pk}, labels: {self.target2label}')

    def encode_samples(self, s_):
        return np.array([self.net.encode(s) for s in s_], ndmin=2)
    
    def make_pairs(self, groups):
        for group in groups:
            if len(self.data[group]['samples']) > 0:
                self.x[group] = torch.from_numpy(self.data[group]['encodings'].reshape(-1, self.data[group]['encodings'].shape[1]).astype('float32'))
                self.y[group] = torch.tensor(self.data[group]['targets'])
            else:
                self.x[group] = torch.tensor([])
                self.y[group] = torch.tensor([])

    def add(self, sample, label):
        if label not in self.labels:
            self.data['knowledge']['samples'].append(sample)
            self.data['knowledge']['labels'].append(label)
            self.init_data()
            self.net.reinit_model(keep_weights=True)

        if sample not in self.data['train']['samples']:
            self.add_to_group('train', sample, label)
            self.pt = len(self.data['train']['samples'])
            print(f'Sample {sample} added to train. pt = {self.pt}')
        
        if sample not in self.data['dev']['samples']:
            self.add_to_group('dev', sample, label)
            print(f'Sample {sample} added to dev.')

        self.make_pairs(('dev', 'train'))

    def add_to_group(self, group, sample, label):
        if len(self.data[group]['samples']) > 0:
            self.data[group]['encodings'] = np.concatenate((self.data[group]['encodings'], self.encode_samples([sample])), axis=0)
        else:
            self.data[group]['encodings'] = self.encode_samples([sample])

        self.data[group]['samples'].append(sample)
        self.data[group]['labels'].append(label)
        self.data[group]['targets'].append(self.labels.index(label))
    
    def loader(self, group, batch_size=16):
        return DataLoader(dataset=I2LDataset(
            x=self.x[group],
            y=self.y[group],
            samples=self.data[group]['samples'],
            labels=self.data[group]['labels']
        ), batch_size=batch_size, shuffle=True)
    

def load_intel_data(dataset_path):
    data = {
        'train': {'samples': [], 'labels': []},
        'dev': {'samples': [], 'labels': []},
        'test': {'samples': [], 'labels': []},
    }

    lims = {'train': 50, 'test': 300, 'dev': 20}

    for group in data.keys():
        for img_path in glob(join(dataset_path, group, '*', '*.jpg')):
            label = img_path.split('/')[-2]
            
            if data[group]['labels'].count(label) < lims[group]:
                data[group]['samples'].append(img_path)
                data[group]['labels'].append(label)

        print(f'# Intel {group} samples: {len(data[group]["samples"])}')

    return data

class IntelData:

    def __init__(self, net, data):
        self.net = net
        self.data = data

        self.labels = None
        self.target2label = None
        self.n = None
        self.m = None
        self.pt = None
        self.pk = None
        self.x = {'train': None, 'dev': None, 'test': None}
        self.y = {'train': None, 'dev': None, 'test': None}
        self.init_data()

    def init_data(self):
        self.labels = sorted(list(set(self.data['test']['labels'])))
        self.target2label = {target:label for target, label in enumerate(self.labels)}

        self.data['train']['targets'] = [self.labels.index(label) for label in self.data['train']['labels']]
        self.data['dev']['targets'] = [self.labels.index(label) for label in self.data['dev']['labels']]
        self.data['test']['targets'] = [self.labels.index(label) for label in self.data['test']['labels']]
        
        self.data['train']['encodings'] = self.encode_samples(self.data['train']['samples'])
        self.data['dev']['encodings'] = self.encode_samples(self.data['dev']['samples'])
        self.data['test']['encodings'] = self.encode_samples(self.data['test']['samples'])

        self.pt = len(self.data['train']['samples'])
        self.pk = len(self.data['test']['samples'])

        self.make_pairs(('test', 'dev', 'train'))

        self.n = len(self.x['test'][0])
        self.m = len(self.labels)

        print(f'Data initialized: n = {self.n}, m = {self.m}, pt = {self.pt}, pk = {self.pk}, labels: {self.target2label}')

    def encode_samples(self, s_):
        return np.array([self.net.encode(s) for s in s_], ndmin=2)
    
    def make_pairs(self, groups):
        for group in groups:
            if len(self.data[group]['samples']) > 0:
                self.x[group] = torch.from_numpy(self.data[group]['encodings'].reshape(-1, self.data[group]['encodings'].shape[1]).astype('float32'))
                self.y[group] = torch.tensor(self.data[group]['targets'])
            else:
                self.x[group] = torch.tensor([])
                self.y[group] = torch.tensor([])

    def loader(self, group, batch_size=16):
        return DataLoader(dataset=I2LDataset(
            x=self.x[group],
            y=self.y[group],
            samples=self.data[group]['samples'],
            labels=self.data[group]['labels']
        ), batch_size=batch_size, shuffle=True)
    