import os
import numpy as np
import pandas as pd
from glob import glob
from pre_utils import argument_setting, stroke_statistics


def _csv2npy(csv_root: str, keep=False):
    """
    convert augmented csv file to npy file

    Args:
        csv_root (str): root of csv file path
        keep (bool): keep original data. Defaults to False.
    """
    from pathlib import Path
    from tqdm import tqdm

    for csv_file in tqdm(Path(csv_root).glob('**/*.csv'), desc='Converting'):
        csv_arr = pd.read_csv(csv_file, header=None).to_numpy()
        np.save(os.path.splitext(csv_file)[0], csv_arr)

        if keep == False:
            csv_file.unlink()


def _npy2csv(npy_root: str, keep=False):
    """
    convert augmented npy file to csv file

    Args:
        npy_root (str): root of npy file path
        keep (bool): keep original data. Defaults to False.
    """
    from pathlib import Path
    from tqdm import tqdm

    for npy_file in tqdm(Path(npy_root).glob('**/*.npy'), desc='Inverse converting'):
        # load npy file
        npy_df = pd.DataFrame(np.load(npy_file, allow_pickle = True))

        # save as csv file
        npy_df.to_csv(f'{os.path.splitext(npy_file)[0]}.csv', header=False, index=False)

        if keep == False:
            npy_file.unlink()


def addnoise(target_data, noise):
    """add noise in range [noise[0], noise[1]]

    Args:
        target_data (pandas.Dataframe): the data to be added noise
        noise ([float, float]): the range of noise to add

    Returns:
        pandas.Dataframe: the data after added noise
    """
    random_data = pd.DataFrame(
        np.random.uniform(noise[0], noise[1], size=(target_data.shape[0], target_data.shape[1] - 1))
    )
    random_data[6] = [""] * random_data.shape[0]
    data = target_data.reset_index(drop=True).add(random_data)

    return data

def extended_length(data, stroke_len, mode):
    """add last line to len of stroke_len.

    Args:
        data (pd.dataframe): data to be added last line
        stroke_len (int): the len of each stroke
        mode (string): the mode of the extended method

    Returns:
        pd.dataframe: the data after added last line
    """

    if mode == 'tail' or data.shape[0] == 1:
        new_data = data.append(data.iloc[[-1] * (stroke_len - data.shape[0])], ignore_index=True)

    # Interpolation
    elif mode == 'inter':
        new_data = data
        index = 0
        while new_data.shape[0] < stroke_len:
            mid = new_data.iloc[index,:].copy()
            mid[:-1] = (new_data.iloc[index, :-1] + new_data.iloc[index + 1, :-1]) / 2
            tmp = pd.concat([new_data[:index + 1], mid.to_frame().transpose()], ignore_index=True)
            new_data = pd.concat([tmp, new_data[index + 1:]], ignore_index=True)
            index += 2
            if index >= new_data.shape[0] - 1:
                index = 0

    return new_data

def csv_parser(char_num, txt_name, target_path, train_path, stroke_len, args):
    """split txt file to csv file by stroke

    Version 6:
        save as npy

    Args:
        char_num (string): the number of the character
        txt_name (string): the txt file name of the character
        target_path (string): the path of target data folder
        train_path (string): the path of train data folder
        args : argument

    Returns:
        int: the number of strokes
    """

    # if not os.path.exists(f'{target_path}/{char_num}'):
    #     os.mkdir(f'{target_path}/{char_num}')
    if not os.path.exists(f'{train_path}/{char_num}'):
        os.mkdir(f'{train_path}/{char_num}')

    data = pd.read_table(txt_name, header=None, sep=' ')    # read txt file to pandas dataframe
    data.drop(columns=[0, 1, 8], inplace=True)              # 把不用用到的column砍掉
    data.columns = range(data.shape[1])                     # 重新排列 column
    stroke_total = len(data.groupby(data.iloc[:, -1]).groups) # 總筆畫

    for stroke_num in range(stroke_total):
        stroke_num = stroke_num + 1         # int
        stroke_idx = f'{stroke_num:0{args.stroke_idx}d}'    # string

        # make directory
        # if not os.path.exists(f'{target_path}/{char_num}/{stroke_idx}'):
        #     os.mkdir(f'{target_path}/{char_num}/{stroke_idx}')
        if not os.path.exists(f'{train_path}/{char_num}/{stroke_idx}'):
            os.mkdir(f'{train_path}/{char_num}/{stroke_idx}')

        # index of each stroke
        each_stroke = data.groupby(data.iloc[:, -1]).groups.get(f'stroke{stroke_num}')
        target_data = data.iloc[each_stroke, :]

        # extend target stroke
        target_data = extended_length(target_data, stroke_len, args.extend)

        # build training data
        for train_num in range(args.train_num):
            filename = f'{char_num}_{stroke_idx}_{train_num + 1 + args.train_start_num:0{args.num_idx}d}.csv'
            file_path = f'{train_path}/{char_num}/{stroke_idx}'
            train_data = addnoise(target_data, args.noise)

            """ # store training data
            train_data.to_csv(f'{file_path}/{filename}', header=False, index=False) """

            # store as npy
            np.save(f'{file_path}/{os.path.splitext(filename)[0]}', train_data.to_numpy())

        """ # store target data
        target_data.to_csv(
            f'{target_path}/{char_num}/{stroke_idx}/{char_num}_{stroke_idx}.csv',
            header=False, index=False
        ) """
        
        # store as npy
        # np.save(f'{target_path}/{char_num}/{stroke_idx}/{char_num}_{stroke_idx}', target_data.to_numpy())
        
    return stroke_total

def get_less_char(file_list, less_char, total_char):

    new_file_list = file_list[0:total_char:int(total_char/less_char)]
    max_len = stroke_statistics(new_file_list)

    return new_file_list, max_len

def preprocessor(args):

    file_list = sorted(glob(os.path.join(args.input_path, '*.txt')))
    stroke_len = args.stroke_length
    if args.less == True:
        # file_list, stroke_len = get_less_char(file_list, args.less_char, args.total_char)
        file_list, _ = get_less_char(file_list, args.less_char, args.total_char)

    # build training data
    if args.test_char == None:
        print('Build training data ...\n')
        print(f'input path = {args.input_path}')
        target_path = f'{args.root_path}/{args.target_path}/'
        train_path = f'{args.root_path}/{args.train_path}/'
        if not os.path.exists(args.root_path):
            os.mkdir(args.root_path)
        # if not os.path.exists(target_path):
        #     os.mkdir(target_path)
        if not os.path.exists(train_path):
            os.mkdir(train_path)

        # count the number of the strokes
        stroke_count = 0

        for txt_name in file_list:
            char_num = txt_name.split('_')[0][-1 * args.char_idx:]
            stroke_add = csv_parser(char_num, txt_name, target_path, train_path, stroke_len, args)
            stroke_count += stroke_add
            print(f'{char_num} finished ...')

        print(f'\nroot path = {args.root_path}')
        print(f'target path = {target_path}')
        print(f'train path = {train_path}')
        print(f'there are {stroke_count} strokes in all target data.')
        print(f'Each stroke length = {stroke_len}.')

    # build testing data
    else:
        char_num = f'{args.test_char:0{args.char_idx}d}'
        test_path = args.test_path

        if not os.path.exists(args.root_path):
            os.mkdir(args.root_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        test_char_path = f'{test_path}/{char_num}/'

        # print(f'Building {char_num} testing data ...\n')
        # print(f'input path = {args.input_path}')
        # print(f'root path = {args.root_path}')
        # print(f'test path = {test_path}\n')

        if not os.path.exists(test_char_path):
            os.mkdir(test_char_path)
        # print(f'Build the director {test_char_path} success ...\n')

        txt_name = f'{args.input_path}/char0{char_num}_stroke.txt'
        data = pd.read_table(txt_name, header=None, sep=' ')        # read txt file to pandas dataframe
        data.drop(columns=[0, 1, 8], inplace=True)                  # 把不用用到的column砍掉
        data.columns = range(data.shape[1])                         # 重新排列 column
        stroke_total = len(data.groupby(data.iloc[:, -1]).groups)   # 總筆畫

        for stroke_num in range(stroke_total):
            stroke_num = stroke_num + 1         # int
            stroke_idx = f'{stroke_num:0{args.stroke_idx}d}'    # string

            # make testing directory
            if not os.path.exists(f'{test_char_path}/{stroke_idx}'):
                os.mkdir(f'{test_char_path}/{stroke_idx}')

            # index of each stroke
            each_stroke = data.groupby(data.iloc[:, -1]).groups.get(f'stroke{stroke_num}')
            target_data = data.iloc[each_stroke, :]

            # build training data
            for test_num in range(args.test_num):
                filename = f'{char_num}_{stroke_idx}_{test_num + 1:0{args.num_idx}d}.csv'

                # add test data last line
                target_data = extended_length(target_data, stroke_len, args.extend)
                test_data = addnoise(target_data, args.noise)

                # store target data
                # if not os.path.exists(f'{target_path}/{char_num}'):
	            #     os.mkdir(f'{target_path}/{char_num}')
                # np.save(f'{target_path}/{char_num}/{stroke_idx}/{char_num}_{stroke_idx}', target_data.to_numpy())
                # store training data
                """ test_data.to_csv(f'{test_char_path}{stroke_idx}/{filename}', header=False ,index=False) """
                np.save(f'{test_char_path}{stroke_idx}/{os.path.splitext(filename)[0]}', test_data.to_numpy())

        print(f'Build {char_num} testing data finished ...\n')

    print('Pre-Processing Done!!!')

if __name__ == '__main__':
    args = argument_setting()

    # convert root path
    if args.convert:
        _csv2npy(args.root_path)
        os._exit(0)
    
    if args.inverse:
        _npy2csv(args.root_path)
        os._exit(0)

    preprocessor(args)