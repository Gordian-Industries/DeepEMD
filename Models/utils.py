import os
import shutil
import time
import pprint
import torch
import numpy as np
import os.path as osp
import random
from torchvision import transforms
from PIL import Image

def save_list_to_txt(name,input_list):
    f=open(name,mode='w')
    for item in input_list:
        f.write(item+'\n')
    f.close()

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print ('use gpu:',gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()





def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print ('create folder:',path)
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()




_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm



def load_model(model,dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']
    if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    model.load_state_dict(model_dict)

    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def detect_grad_nan(model):
    for param in model.parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()

def get_transformation_format():
    image_size = 84
    transform = transforms.Compose([
        transforms.Resize([92, 92]),
        transforms.CenterCrop(image_size),

        transforms.ToTensor(),
        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    return transform

def get_transformed_images(path, transform):
    images = []
    images_path = os.listdir(path)
    for image_path in images_path:
        try:
            images.append(transform(Image.open(f"{path}/{image_path}").convert('RGB')))
        except:
            os.remove(f"{path}/{image_path}")

    return images


def get_data(query_path, reference_path):
    transform = get_transformation_format()

    q_image = get_transformed_images(query_path, transform)
    reference_images = get_transformed_images(reference_path, transform)

    return torch.stack(q_image).cuda(), torch.stack(reference_images).cuda()


def top_five_indexes(arr):
    if len(arr) < 5:
        return "Array length is less than 5"

    top_indexes = list(range(5))
    top_values = arr[:5]

    for i, num in enumerate(arr[5:], start=5):
        min_index = top_values.index(min(top_values))
        if num > top_values[min_index]:
            top_values[min_index] = num
            top_indexes[min_index] = i

    return top_indexes






