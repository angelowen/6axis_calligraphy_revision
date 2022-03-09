from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os

def argument_setting():
    r"""
    return the arguments
    """
    parser = ArgumentParser()
    parser.add_argument('--train-num', type=int, default=100,
                        help='set the numbers of the training datas in each stroke you want to create (default: 100)')
    parser.add_argument('--train-start-num', type=int, default=0,
                        help='set the start number of the training datas (default: 0)')
    parser.add_argument('--noise', type=float, nargs=2, default=[-1,1],
                        help='set the noise range (default: [-1, 1])')
    parser.add_argument('--stroke-length', type=int, default=150,
                        help='set the length of each stroke (default: 150)')
    
    parser.add_argument('--test-char', type=int, default=None,
                        help='set the character number of the testing data you want to build (default: None)')
    parser.add_argument('--test-num', type=int, default=30,
                        help='set the numbers of the testing datas you want to create (default: 30)')
    
    parser.add_argument('--char-idx', type=int, default=4,
                        help='set the index length of each char of file name (default: 4)')
    parser.add_argument('--stroke-idx', type=int, default=2,
                        help='set the length of each stroke (default: 2)')
    parser.add_argument('--num-idx', type=int, default=4,
                        help='set the length of each stroke (default: 4)')


    parser.add_argument('--input-path', type=str, default='/home/jefflin/6axis/',
                        help='set the path of the original datas (default: /home/jefflin/6axis/)')
    parser.add_argument('--root-path', type=str, default='./dataset/',
                        help='set the root path (default: ./dataset/)')
    parser.add_argument('--target-path', type=str, default='./target/',
                        help='set the path of the target datas (default: ./target/)')
    parser.add_argument('--train-path', type=str, default='./train/',
                        help='set the path of the training datas (default: ./train/)')
    parser.add_argument('--test-path', type=str, default='./test/',
                        help='set the path of the testing datas (default: ./dataset/test/)')

    # for extended length
    parser.add_argument('--extend', type=str, default='tail',
                        metavar='tail, inter', help="set the complement method (default: 'tail')")

    # for testing dataset
    parser.add_argument('--less', action='store_true', default=False,
                        help='get the less of the dataset (default: False)')
    parser.add_argument('--less-char', type=int, default=100,
                        help='set the numbers of the training characters you want to create (default: 100)')
    parser.add_argument('--total-char', type=int, default=900,
                        help='set the numbers of the total training characters (default: 900)')

    # convert
    parser.add_argument('--convert', action='store_true', default=False,
                        help='convert csv to npy file (default: False)')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='convert npy to csv file (default: False)')
    parser.add_argument('--keep', action='store_true', default=False,
                        help='keep original data while converting or inversing (default: False)')


    return parser.parse_args()

def stroke_statistics(file_list, mode='max'):
    r""" 
    parameter:
    path: path of the 6d axis csv file
    mode: output statstic. i.e. mode:max means return the maximum stroke number
    count the mean, maximum, minimum of the stroke statistics
    output: use parameter 
    """
    max_cnt = np.int64(0)
    mean_cnt = np.int64(0)
    min_cnt = np.int64(100)

    for txt_name in file_list:
        df = pd.read_csv(txt_name, header=None, sep=' ')
        tmp = df.groupby(9)[0].count()

        if tmp.max() > max_cnt:
            max_cnt = tmp.max()

        if tmp.mean() > mean_cnt:
            mean_cnt = tmp.mean()

        if tmp.min() < min_cnt:
            min_cnt = tmp.min()

    # print(f'max stroke number:{max_cnt}\nmean stroke number:{mean_cnt}\nmin stroke number:{min_cnt}')

    return {
        'max' : max_cnt,
        'mean': mean_cnt,
        'min' : min_cnt
    }.get(mode, 'error')
