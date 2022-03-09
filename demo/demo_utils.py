import sys
sys.path.append('..')
import time
import pandas as pd
from argparse import ArgumentParser
from utils import StorePair

def argument_setting():
    r"""
    return the arguments
    """
    parser = ArgumentParser()

    #################################
    ## demo process setting
    parser.add_argument('--char-max', type=int, default=900,
                        help='set the upper limit of characters number (default: 900)')
    parser.add_argument('--noise-max', type=float, default=10,
                        help='set the maximun amplitude of noise range (default: 10)')
    parser.add_argument('--demo-post', action='store_true', default=True,
                        help='Just post-process the demo required files (default: True)')

    #################################
    ## dataset setting
    parser.add_argument('--noise', type=float, nargs=2, default=[-1,1],
                        help='set the noise range (default: [-1, 1])')
    parser.add_argument('--stroke-length', type=int, default=150,
                        help='set the length of each stroke (default: 150)')
    
    parser.add_argument('--test-char', type=int, default=None,
                        help='set the character number of the testing data you want to build (default: None)')
    parser.add_argument('--test-num', type=int, default=1,
                        help='set the numbers of the testing datas you want to create (default: 1)')

    # extended length method
    parser.add_argument('--extend', type=str, default='inter',
                        metavar='tail, inter', help="set the complement method (default: 'inter')")

    # dataset file name format
    parser.add_argument('--char-idx', type=int, default=4,
                        help='set the index length of each char of file name (default: 4)')
    parser.add_argument('--stroke-idx', type=int, default=2,
                        help='set the length of each stroke (default: 2)')
    parser.add_argument('--num-idx', type=int, default=4,
                        help='set the length of each stroke (default: 4)')

    # dataset path
    parser.add_argument('--root-path', type=str, default='../dataset/',
                        help='set the root path (default: ../dataset/)')
    parser.add_argument('--input-path', type=str, default='../dataset/6axis/',
                        help='set the path of the original datas (default: ../dataset/6axis/)')
    parser.add_argument('--test-path', type=str, default='../dataset/test/',
                        help='set the path of the testing datas (default: ../dataset/test/)')
    parser.add_argument('--target-path', type=str, default='../dataset/target/',
                        help="target dataset path (default: '../dataset/target/')")

    # for preprocessor setting
    parser.add_argument('--less', action='store_true', default=False,
                        help='get the less of the dataset (default: False)')

    ##########################
    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='set the logger path of pytorch model (default: ./logs)')
    # save setting
    parser.add_argument('--save-path', type=str, default='./output/',
                        help='set the output file (csv or txt) path (default: ./output/)')

    # usb path
    parser.add_argument('--usb-path', type=str,
                        help='set the USB path to copy to (default: None)')

    ##########################
    # testing args
	# doc setting
    parser.add_argument('--doc', type=str, metavar='../doc/sample.yaml',
                        help='load document file by position(default: None)')

    # dataset setting
    parser.add_argument('--batch-size', type=int, default=64,
                        help='set the batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')

    # model setting
    parser.add_argument('--model-name', type=str, default='FSRCNN',
                        metavar='FSRCNN, DDBPN' ,help="set model name (default: 'FSRCNN')")
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')
    parser.add_argument('--model-args', action=StorePair, nargs='+', default={},
                        metavar='key=value', help="set other args (default: {})")
    parser.add_argument('--load', action='store_false', default=True,
                        help='load model parameter from exist .pt file (default: True)')
    parser.add_argument('--version', type=int, dest='load',
                        help='load specific version (default: False)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')

    # loss setting
    parser.add_argument('--alpha', type=float, default=1,
                        help="set loss 1's weight (default: 1)")
    parser.add_argument('--beta', type=float, default=1,
                        help="set loss 2's weight (default: 1)")
    parser.add_argument('--criterion', type=str, default='huber',
                        help="set criterion (default: 'huber')")

    # performance setting
    parser.add_argument('--timer', default=False, action='store_true',
                        help='compute execution time function by function (default: False)')
    parser.add_argument('--content-loss', default=False, action='store_true',
                        help='compute content loss (default: False)')
    parser.add_argument('--nonefficient', default=False, action='store_true',
                        help='improve demo execution time (default: False)')
    parser.add_argument('--gui', default=False, action='store_true',
                        help='Demo with gui (default: False)')
    parser.add_argument('--combine', default=False, action='store_true',
                        help='Combine input and output txt file(default: False)')

    return parser.parse_args()


def timer(func):
    def wrapper(*args, **kwargs):
        
        print(f'evaluate "{func.__name__}" execution time')
        start = time.time()

        func(*args, **kwargs)

        end = time.time()
        print(f'"{func.__name__}" execution time: {end - start:.2f}')

        # return execution time
        return end - start

    return wrapper

def getmidxy(input_path, char_num):

    # python -c "from demo_utils import getmidxy; getmidxy('./dataset/6axis/', 42)"
    # Results:
    #    Datun X = -31.784749999999995
    #    Datum Y = 366.56565

    char_num = f'{char_num:04d}'
    txt_name = f'{input_path}/char0{char_num}_stroke.txt'
    data = pd.read_table(txt_name, header=None, sep=' ')  # read txt file to pandas dataframe

    mid_x = (data[2].max() + data[2].min()) / 2
    mid_y = (data[3].max() + data[3].min()) / 2
    print(f'Datun X = {mid_x}\nDatum Y = {mid_y}')

    return mid_x, mid_y

def translation(source):
    """Translation to datum position

    Args:
        source (string): the source file to processing
    """

    # value from getmidxy('./dataset/6axis/', 42)
    datum_x = -31.78475
    datum_y = 366.56565

    # read source file to pandas dataframe
    data = pd.read_table(source, header=None, sep=' ')

    # get displacement
    disp_x = datum_x - (data[2].max() + data[2].min()) / 2
    disp_y = datum_y - (data[3].max() + data[3].min()) / 2

    # translation
    data[2] = data[2].add(disp_x)
    data[3] = data[3].add(disp_y)

    return data

def combine2txt(input, output):

    input[2] = input[2].sub(60)
    data = input.append(output, ignore_index=True)

    return data

def demo_post(args):

    test_all_output = translation(f'{args.save_path}/test_char/test_all_output.txt')

    if args.combine:
        test_all_input = translation(f'{args.save_path}/test_char/test_all_input.txt')

        # combine input and output txt files
        demo_data = combine2txt(test_all_input, test_all_output)

    else:
        demo_data = test_all_output

    # initial position
    init_pos = pd.DataFrame([['movl', '0', '-120', '330', '440', '175', '0', '180', '100.0']])
    init_pos[9] = demo_data.iloc[-1, 9]
    demo_data = demo_data.append(init_pos, ignore_index=True)

    # store to USB
    demo_data.to_csv(f'{args.usb_path}/demo_output.txt', header=False, index=False, sep=' ')

    print('Demo Postprocessing Finished!!!')