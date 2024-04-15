import argparse

import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from Models.dataloader.samplers import CategoriesSampler
from Models.models.Network import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *

DATA_DIR='datasets/'
# DATA_DIR='/home/zhangchi/dataset'
MODEL_DIR='deepemd_trained_model/miniimagenet/fcn/max_acc.pth'



parser = argparse.ArgumentParser()
# about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=2)  # number of query image per class
parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet_yao','cifar_fs'])
parser.add_argument('-set', type=str, default='test', choices=['train','val', 'test'])
# about model
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-metric', type=str, default='cosine', choices=[ 'cosine' ])
parser.add_argument('-norm', type=str, default='center', choices=[ 'center'])
parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
#deepemd fcn only
parser.add_argument('-feature_pyramid', type=str, default=None)
#deepemd sampling only
parser.add_argument('-num_patch',type=int,default=9)
#deepemd grid only patch_list
parser.add_argument('-patch_list',type=str,default='2,3')
parser.add_argument('-patch_ratio',type=float,default=2)
# solver
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
# SFC
parser.add_argument('-sfc_lr', type=float, default=100)
parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=100)
parser.add_argument('-sfc_bs', type=int, default=4)
# others
parser.add_argument('-test_episode', type=int, default=5000)
parser.add_argument('-gpu', default='0,1')
parser.add_argument('-data_dir', type=str, default=DATA_DIR)
parser.add_argument('-model_dir', type=str, default=MODEL_DIR)
parser.add_argument('-seed', type=int, default=1)


args = parser.parse_args()
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

pprint(vars(args))
set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset=set_up_datasets(args)


# model
model = DeepEMD(args)
model = load_model(model, args.model_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

# test dataset
test_set = Dataset(args.set, args)
sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler, num_workers=0, pin_memory=True)
tqdm_gen = tqdm.tqdm(loader)

# label of query images
ave_acc = Averager()
test_acc_record = np.zeros((args.test_episode,))
label = torch.arange(args.way).repeat(args.query)
label = label.type(torch.cuda.LongTensor)

query_path, ref_path = "data/Corner/query_images", "data/Corner/reference_images"
output_path = "../data/Corner/outputs"

query_images, r_imgs = get_data(query_path, ref_path)
images_path = os.listdir(ref_path)
q_images = os.listdir(query_path)
r_data = None
batch_size = 5
with torch.no_grad():
    for indexes in range(len(q_images)):
        q_img = query_images[indexes,:,:,:]
        q_img = q_img.unsqueeze(0)
        # print(query_images.shape)
        model.module.mode = 'encoder'
        q_data = model(q_img)
        if r_data is None:
            reference_images = []
            logits_list = []
            for start_range in  range(0, r_imgs.shape[0], batch_size):
                if r_imgs.shape[0] > start_range+batch_size:
                    model.module.mode = 'encoder'
                    data = model(r_imgs[start_range:start_range+batch_size, :, :, :])
                    reference_images.append(data)
                    model.module.mode = 'meta'
                    data = model((q_data.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data))
                    data = torch.squeeze(data, dim=1)
                    logits_list.append(data)
                else:
                    model.module.mode = 'encoder'
                    data = model(r_imgs[start_range:, :, :, :])
                    reference_images.append(data)
                    model.module.mode = 'meta'
                    data = model((q_data.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data))
                    data = torch.squeeze(data, dim=1)
                    logits_list.append(data)
                    # reference_images.append(model(r_imgs[start_range:, :, :, :]))
            # r_data = torch.cat(reference_images, dim=0)
            logits = torch.cat(logits_list, dim=0)


        # model.module.mode = 'meta'
        # logits = model((q_data.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), r_data))
        # logits = torch.squeeze(logits, dim=1)
        # print(logits.shape)
        # print(logits.shape)
        probabilities = torch.softmax(logits, dim=0)
        # print(probabilities.shape)
        top5_probabilities, top5_indices = torch.topk(probabilities, k=5)
        top5_indices = top5_indices.tolist()
        print(top5_indices,top5_probabilities)
        # print(logits)
        # pred = torch.argmax(logits, dim=0)
        idx_out = len(os.listdir(output_path))

        for id, idx in enumerate(top5_indices):
            # print(images_path, idx, idx_out, id)
            if not os.path.exists(f"{output_path}/output_000{idx_out}"):
                os.mkdir(f"{output_path}/output_000{idx_out}")
            shutil.copy2(f"{ref_path}/{images_path[idx]}",f"{output_path}/output_000{idx_out}/top_{id+1}.jpg")
            shutil.copy2(f"{query_path}/{q_images[indexes]}", f"{output_path}/output_000{idx_out}/query_image.jpg")
        # print("Prediction is ",idx, images_path[idx])






# with torch.no_grad():
#     for i, batch in enumerate(tqdm_gen, 1):
#         data, _ = [_.cuda() for _ in batch]
#         k = args.way * args.shot
#         model.module.mode = 'encoder'
#         data = model(data)
#         data_shot, data_query = data[:k], data[k:]  # shot: 5,3,84,84  query:75,3,84,84
#         model.module.mode = 'meta'
#         if args.shot > 1:
#             data_shot = model.module.get_sfc(data_shot)
#         logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
#         acc = count_acc(logits, label) * 100
#         ave_acc.add(acc)
#         test_acc_record[i - 1] = acc
#         m, pm = compute_confidence_interval(test_acc_record[:i])
#         tqdm_gen.set_description('batch {}: This episode:{:.2f}  average: {:.4f}+{:.4f}'.format(i, acc, m, pm))
#
#     m, pm = compute_confidence_interval(test_acc_record)
#     result_list = ['test Acc {:.4f}'.format(ave_acc.item())]
#     result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
#     print(result_list[0])
#     print(result_list[1])
