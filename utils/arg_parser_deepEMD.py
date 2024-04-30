import argparse


def parse_arguments(DATA_DIR='datasets/', MODEL_DIR= 'DeepEMD/deepemd_trained_model/miniimagenet/fcn/max_acc.pth'):

    parser = argparse.ArgumentParser()
    # about task
    parser.add_argument('-way', type=int, default=5)
    parser.add_argument('-shot', type=int, default=1)
    parser.add_argument('-query', type=int, default=2)  # number of query image per class
    parser.add_argument('-dataset', type=str, default='miniimagenet',
                             choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs'])
    parser.add_argument('-set', type=str, default='test', choices=['train', 'val', 'test'])
    # about model
    parser.add_argument('-temperature', type=float, default=12.5)
    parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
    parser.add_argument('-norm', type=str, default='center', choices=['center'])
    parser.add_argument('-deepemd', type=str, default='fcn', choices=['fcn', 'grid', 'sampling'])
    # deepemd fcn only
    parser.add_argument('-feature_pyramid', type=str, default=None)
    # deepemd sampling only
    parser.add_argument('-num_patch', type=int, default=9)
    # deepemd grid only patch_list
    parser.add_argument('-patch_list', type=str, default='2,3')
    parser.add_argument('-patch_ratio', type=float, default=2)
    # solver
    parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
    # SFC
    parser.add_argument('-sfc_lr', type=float, default=100)
    parser.add_argument('-sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('-sfc_update_step', type=float, default=100)
    parser.add_argument('-sfc_bs', type=int, default=4)
    # others
    parser.add_argument('-test_episode', type=int, default=5000)
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-data_dir', type=str, default=DATA_DIR)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    return args