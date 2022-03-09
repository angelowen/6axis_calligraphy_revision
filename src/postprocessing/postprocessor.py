import os
from glob import glob
import re
import pandas as pd

from stroke2char import stroke2char
from axis2img import axis2img
from csv2txt import csv2txt
from post_utils import get_len, argument_setting

def postprocessor_dir(dir_path, csv_list, path, args):
    """Do postprocessor to the directory

    Args:
        dir_path (string): the directory path
        csv_list (list of string): the list of csv files
    """

    # initialize for test files
    test_target = pd.DataFrame(None)
    test_input = pd.DataFrame(None)
    test_output = pd.DataFrame(None)

    if len(csv_list) % 3 != 0:
        print(f"Error!!! csv file numbers in {dir_path} can't be divided by 3 !!!")
        return

    # the total number of different files
    file_len = int(len(csv_list) / 3)

    # Build directory
    if file_len != 0:
        if not os.path.exists(f'{dir_path}/test_char/'):
            os.mkdir(f'{dir_path}/test_char/')
        if args.demo_post is False and not os.path.exists(f'{dir_path}/txt/'):
            os.mkdir(f'{dir_path}/txt/')
        if args.demo_post is False and not os.path.exists(f'{dir_path}/pic/'):
            os.mkdir(f'{dir_path}/pic/')

    for file_idx in range(file_len):
	
        # get feature of file name
        file_feature = re.split(r'_input.csv', csv_list[file_idx * 3])[0]

        target = pd.read_csv(os.path.join(dir_path, f'{file_feature}_target.csv'), header=None)

        # get original length
        stroke_len = get_len(target)

        # drop the extra rows and read input and output data
        target.drop(target.index[stroke_len:], inplace=True)
        input = pd.read_csv(
                    os.path.join(dir_path, f'{file_feature}_input.csv'),
                    nrows=stroke_len,
                    header=None,
                )
        output = pd.read_csv(
                    os.path.join(dir_path, f'{file_feature}_output.csv'),
                    nrows=stroke_len,
                    header=None
                )

        # test stroke2char
        if file_feature[:5] == 'test_' and file_feature != 'test_all':
            (
                test_target, test_input, test_output
            ) = stroke2char(
                    target.iloc[:, :-1], input.iloc[:, :-1], output.iloc[:, :-1],
                    test_target, test_input, test_output,
                    dir_path, stroke_len, int(file_feature[5:])
                )

        if args.demo_post is False:
            # axis2img
            axis2img(target, input, output, file_feature, f'{dir_path}/pic/')

            # csv2txt
            csv2txt(target, os.path.join(f'{dir_path}/txt/', f'{file_feature}_target.txt'))
            csv2txt(input, os.path.join(f'{dir_path}/txt/', f'{file_feature}_input.txt'))
            csv2txt(output, os.path.join(f'{dir_path}/txt/', f'{file_feature}_output.txt'))

    # save test char file
    if test_target.shape[0] != 0:

        if args.input_path == None:
            input_path = '/home/jeff/6axis/'
        else:
            input_path = args.input_path

        # 採用刪除法，較快
        drop_list = compare(test_target.round(4), dir_path, input_path, args)
        test_target, test_input, test_output = inverse_len(test_target, test_input, test_output, drop_list)

        if args.demo_post is False:
            test_target.to_csv(os.path.join(dir_path, 'test_all_target.csv'), header=False, index=False)
            test_input.to_csv(os.path.join(dir_path, 'test_all_input.csv'), header=False, index=False)
            test_output.to_csv(os.path.join(dir_path, 'test_all_output.csv'), header=False, index=False)

        # axis2img
        axis2img(test_target, test_input, test_output, 'test_all', f'{dir_path}/test_char/')

        # csv2txt
        csv2txt(test_target, os.path.join(f'{dir_path}/test_char/', f'test_all_target.txt'))
        csv2txt(test_input, os.path.join(f'{dir_path}/test_char/', f'test_all_input.txt'))
        csv2txt(test_output, os.path.join(f'{dir_path}/test_char/', f'test_all_output.txt'))

def compare(test_target, path, input_path, args):
    """Compare org data and target data， return a list stored the useless row index

    Args:
        test_target (pandas.Dataframe): the data to compare
        path (string): current directory path

    Returns:
        List: a list stored the useless row index
    """
    # 取得字元編號
    char_num = os.path.abspath(path)[-3:]
    if char_num.isdigit() == False:
            char_num = f'{args.test_char:03d}'
    filename = f"{input_path}/char00{char_num}_stroke.txt"

    # 讀入原始 txt 檔，並捨去不需要的 column
    try:
        data_txt = pd.read_csv(filename, sep=" ", header=None).drop(columns=[0, 1, 8])
    except:
        return []

    data_txt.columns = range(data_txt.shape[1])

    j = 0
    drop_list = []
    for i in range(test_target.shape[0]):
        # 比較兩 row 的值是不是不相等
        tmp = test_target.iloc[i,:].eq(data_txt.iloc[j,:])
        if len(tmp.index[tmp]) != 7:
            drop_list.append(i)  # 存入到紀錄要丟棄的 list 裡
        else:  # 移動原始資料的 index
            j += 1
        if j == data_txt.shape[0]:
            break

    return drop_list

def inverse_len(test_target, test_input, test_output, drop_list):
	
    test_target = test_target.drop(drop_list).reset_index(drop=True)
    test_input = test_input.drop(drop_list).reset_index(drop=True)
    test_output = test_output.drop(drop_list).reset_index(drop=True)

    return test_target,test_input,test_output

def postprocessor(args):
    """postprocess output files

    Args:
        path (string, optional): the path of the output directory.
    """
    path = args.save_path

    # check the path exists or not
    if not os.path.exists(path):
        print(f'{path} is not exist!!!')
        return

    for root, _ , files in os.walk(path):

        # get the list of the csv file name
        csv_files = sorted(list(filter(lambda x: re.match(r'(.*).csv', x), files)))

        # postprocess
        postprocessor_dir(root, csv_files, path, args)

        print(f'{root}\tfinished...')

    print('Post-Processing Done!!!')


if __name__ == '__main__':
    args = argument_setting()
    postprocessor(args)