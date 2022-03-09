import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
from model import FeatureExtractor
from utils import  (model_builder, out2csv, model_config, config_loader, StorePair, NormScaler,
    criterion_builder, writer_builder)
from dataset import AxisDataSet

def test_argument(inhert=False):
    """return test arguments

    Args:
        inhert (bool, optional): return parser for compatiable. Defaults to False.

    Returns:
        parser_args(): if inhert is false, return parser's arguments
        parser(): if inhert is true, then return parser
    """

    # for compatible
    parser = ArgumentParser(add_help=not inhert)

    # doc setting
    parser.add_argument('--doc', type=str, metavar='./doc/sample.yaml',
                        help='load document file by position(default: None)')

    # dataset setting
    parser.add_argument('--stroke-length', type=int, default=150,
                        help='control the stroke length (default: 150)')
    parser.add_argument('--test-path', type=str, default='/home/jefflin/dataset/test_all',
                        help="test dataset path (default: '/home/jefflin/dataset/test_all')")
    parser.add_argument('--target-path', type=str, default='/home/jefflin/dataset/target',
                        help="target dataset path (default: '/home/jefflin/dataset/target')")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')
    parser.add_argument('--test-num', type=int, default=30,
                        help='set the number of each stroke in testing dataset (default: 30)')

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

    # test setting
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help="set loss 1's weight (default: 1e-3)")
    parser.add_argument('--beta', type=float, default=1,
                        help="set loss 2's weight (default: 1)")
    parser.add_argument('--criterion', type=str, default='mse',
                        help="set criterion (default: 'mse')")

    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='set the logger path of pytorch model (default: ./logs)')
    # save setting
    parser.add_argument('--save-path', type=str, default='./output',
                        help='set the output file (csv or txt) path (default: ./output)')

    # for the compatiable
    if inhert is True:
        return parser

    return parser.parse_args()


@torch.no_grad()
def demo_eval(model, test_loader, criterion, args, feature_extractor=None):
    """evaluate function for demo only

    Notice that model and feature_extractor must store in cuda before call the function

    Args:
        model (torch.nn): must store in gpu
        test_loader (torch.utils.data.DataLoader): one calligraphy only
        criterion (torch.nn): Defaults to mse loss
        args : defined by demo arguments 
        feature_extractor (torch.nn, optional): Ignore it for improving execution time. Defaults to None.
    """

    model.eval()
    err = 0.0

    # out2csv
    i = 0   # count the number of loops
    j = 0   # count the number of data

    for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
        inputs, target, _ = data
        inputs, target = inputs.cuda(), target.cuda()

        # normalize inputs and target
        # inputs = input_scaler.fit(inputs)

        pred = model(inputs)

        # denormalize
        # inputs = input_scaler.inverse_transform(inputs)
        # pred = input_scaler.inverse_transform(pred)

        # out2csv
        while j - (i * args.batch_size) < pred.size(0):
            out2csv(inputs, f'test_{int(j/args.test_num)+1}', 'input', j - (i * args.batch_size), args.save_path, args.stroke_length, spec_flag=True)
            out2csv(pred, f'test_{int(j/args.test_num)+1}', 'output', j - (i * args.batch_size), args.save_path, args.stroke_length, spec_flag=True)
            out2csv(target, f'test_{int(j/args.test_num)+1}', 'target', j - (i * args.batch_size), args.save_path, args.stroke_length, spec_flag=True)
            j += args.test_num
        i += 1

        # MSE loss
        mse_loss = args.alpha * criterion(pred, target)
        loss = mse_loss

        # compute content loss by vgg features
        if feature_extractor:
            feature_extractor.eval()

            # content loss
            gen_feature = feature_extractor(pred)
            real_feature = feature_extractor(target)
            content_loss = args.beta * criterion(gen_feature, real_feature)

            # add content loss to loss function
            loss += content_loss

        # for compatible
        err += loss.sum().item() * inputs.size(0)

    err /= len(test_loader.dataset)
    print(f'test error:{err:.4f}')


@torch.no_grad()
def test(model, test_loader, criterion, args):
    # set model path
    if args.load is not False:
        _, log_path = writer_builder(
            args.log_path, args.model_name, load=args.load
        )
        model_path = os.path.join(log_path, f'{args.model_name}_{args.scale}x.pt')

    # load model parameters
    # print(model_path)
    checkpoint = torch.load(model_path, map_location=f'cuda:{args.gpu_id}')

    # try-except to compatible
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print('Warning: load older version')
        model.feature = nn.Sequential(*model.feature, *model.bottle)
        model.bottle = nn.Sequential()
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()

    # normalize scaler
    # input_scaler = NormScaler(mean=0.5, std=0.5)

    # declare content loss
    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.eval()

    err = 0.0

    # out2csv
    i = 0   # count the number of loops
    j = 0   # count the number of data

    for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
        inputs, target, _ = data
        inputs, target = inputs.cuda(), target.cuda()

        # normalize inputs and target
        # inputs = input_scaler.fit(inputs)

        pred = model(inputs)

        # denormalize
        # inputs = input_scaler.inverse_transform(inputs)
        # pred = input_scaler.inverse_transform(pred)

        # out2csv
        while j - (i * args.batch_size) < pred.size(0):
            out2csv(inputs, f'test_{int(j/args.test_num)+1}', 'input', j - (i * args.batch_size), args.save_path, args.stroke_length, spec_flag=True)
            out2csv(pred, f'test_{int(j/args.test_num)+1}', 'output', j - (i * args.batch_size), args.save_path, args.stroke_length, spec_flag=True)
            out2csv(target, f'test_{int(j/args.test_num)+1}', 'target', j - (i * args.batch_size), args.save_path, args.stroke_length, spec_flag=True)
            j += args.test_num
        i += 1

        # MSE loss
        mse_loss = args.alpha * criterion(pred, target)

        # content loss
        gen_feature = feature_extractor(pred)
        real_feature = feature_extractor(target)
        content_loss = args.beta * criterion(gen_feature, real_feature)

        # for compatible
        loss = content_loss + mse_loss
        err += loss.sum().item() * inputs.size(0)

    err /= len(test_loader.dataset)
    print(f'test error:{err:.4f}')

if __name__ == '__main__':
    # argument setting
    test_args = test_argument()
    
    if test_args.doc:
        test_args = config_loader(test_args.doc, test_args)
    # config
    model_config(test_args, save=False)     # print model configuration of evaluation

    # set cuda
    torch.cuda.set_device(test_args.gpu_id)

    # model
    model = model_builder(test_args.model_name, test_args.scale, **test_args.model_args).cuda()

    # criteriohn
    criterion = criterion_builder(test_args.criterion)
    # optimizer = None # don't need optimizer in test

    # dataset
    test_set = AxisDataSet(test_args.test_path, test_args.target_path)

    test_loader = DataLoader(test_set,
                             batch_size=test_args.batch_size,
                             shuffle=False,
                             num_workers=test_args.num_workers,
                            #  pin_memory=True,
                             pin_memory=False,
                             )

    # test
    test(model, test_loader, criterion, test_args)
