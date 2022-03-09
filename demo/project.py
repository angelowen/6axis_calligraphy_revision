import os, sys, shutil
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# system path
sys.path.append(r'../src/')
sys.path.append(r'../src/postprocessing')
sys.path.append(r'../src/preprocessing')
sys.path.append(r'../src/model')

# self defined
from demo_utils import (argument_setting, timer, demo_post)
from preprocessing import preprocessor
from postprocessing import (postprocessor, verification)
from model import FeatureExtractor
from eval import test, demo_eval

# test defined
from utils import  (model_builder, model_config, config_loader, criterion_builder)
from dataset import AxisDataSet

# execution statistics
exe_stat = []

def model_env(args):
    """building model environment avoiding to instantiate model.

    Args:
        args : model arguments which is control by demo_utils.argument_setting

    Returns:
        model (torch.nn): build model in cuda device
        criterion(torch.nn): build criterion. Default to mse loss
        extractor(torch.nn): build vgg content loss in cuda device
    """

    if args.doc:
        args = config_loader(args.doc, args)

    # set cuda device
    torch.cuda.set_device(args.gpu_id)

    # model version control
    version = args.load if type(args.load) is int else 0

    # model path and parameter
    model_path = os.path.join(
        args.log_path, args.model_name, f'version_{version}',f'{args.model_name}_{args.scale}x.pt')

    checkpoint = torch.load(model_path, map_location=f'cuda:{args.gpu_id}')

    # loading model
    model = model_builder(args.model_name, args.scale, **args.model_args).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # build criterion
    criterion = criterion_builder(args.criterion)

    # loading feature extractor
    extractor = FeatureExtractor().cuda() if args.content_loss else None

    return model, criterion, extractor


def data_env(args):
    """build data loader

    Args:
        args : model arguments which is control by demo_utils.argument_setting

    Returns:
        data_loader [torch.utils.data.DataLoader]: dataloader for only one character
    """
    dataset = AxisDataSet(args.test_path, args.target_path)

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                            #  pin_memory=True,
                             pin_memory=False,
                            )
    
    return data_loader

@timer
def efficient_demo(args,noise=0,test_char=42):
    # Input the number of the character
    if args.gui:
        args.test_char = test_char
    else:
        test_char = 0
        while test_char < 1 or test_char > args.char_max:
            test_char = int(input(f"Input the number of character (1-{args.char_max}):"))
            if test_char < 1 or test_char > args.char_max:
                print("Please input the number in the correct range!")
            else:
                args.test_char = test_char
                break
    
    # Input the upper limit of the noise range
    if args.gui:
        args.noise = [-1 * noise, noise]
    else:
        noise = -1
        while noise < 0 or noise > args.noise_max:
            noise = float(input(f"Input the upper limit of the noise range (0-{args.noise_max}):"))
            if noise < 0 or noise > args.noise_max:
                print("Please input the number in the correct range!")
            else:
                args.noise = [-1 * noise, noise]
                break
    
    if os.path.exists(args.test_path):
        shutil.rmtree(args.test_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)

    exe_stat.append(
        preprocessor(args)
    )

    print('\n===================================================')

    # construction dataset
    data_loader = data_env(args)

    exe_stat.append(
        demo_eval(args.model, data_loader, args.critetion, args, args.extractor)
    )

    print('\n===================================================')
    exe_stat.append(
        postprocessor(args)
    )

    print('\n===================================================')
    exe_stat.append(
        verification(args)
    )

    if args.usb_path != None:
        print('\n===================================================')
        exe_stat.append(
            demo_post(args)
        )

    print('\n===================================================')
    print(f'Testing number {args.test_char} with noise {args.noise}, Done!!!')


def demo_test(args):
    if args.doc:
        args = config_loader(args.doc, args)
    # config
    # model_config(args, save=False)     # print model configuration of evaluation

    # set cuda
    torch.cuda.set_device(args.gpu_id)

    # model
    model = model_builder(args.model_name, args.scale, **args.model_args).cuda()

    # criteriohn
    criterion = criterion_builder(args.criterion)

    # dataset
    test_set = AxisDataSet(args.test_path, args.target_path)

    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                            #  pin_memory=True,
                             pin_memory=False,
                             )

    # test
    test(model, test_loader, criterion, args)

@timer
def demo(args,noise=0,test_char=42):

    # Input the number of the character
    if args.gui:
        args.test_char = test_char
    else:
        test_char = 0
        while test_char < 1 or test_char > args.char_max:
            test_char = int(input(f"Input the number of character (1-{args.char_max}):"))
            if test_char < 1 or test_char > args.char_max:
                print("Please input the number in the correct range!")
            else:
                args.test_char = test_char
                break
    
    # Input the upper limit of the noise range
    if args.gui:
        args.noise = [-1 * noise, noise]
    else:
        noise = -1
        while noise < 0 or noise > args.noise_max:
            noise = float(input(f"Input the upper limit of the noise range (0-{args.noise_max}):"))
            if noise < 0 or noise > args.noise_max:
                print("Please input the number in the correct range!")
            else:
                args.noise = [-1 * noise, noise]
                break

    if os.path.exists(args.test_path):
        shutil.rmtree(args.test_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)

    exe_stat.append(
        preprocessor(args)
    )

    print('\n===================================================')
    exe_stat.append(
        demo_test(args)
    )

    print('\n===================================================')
    exe_stat.append(
        postprocessor(args)
    )

    print('\n===================================================')
    exe_stat.append(
        verification(args)
    )

    if args.usb_path != None:
        print('\n===================================================')
        exe_stat.append(
            demo_post(args)
        )

    print('\n===================================================')
    print(f'Testing number {args.test_char} with noise {args.noise}, Done!!!')

def demo_main(args, noise=0, word_idx=42):
    # argument setting
    # args = argument_setting()

    # config
    # model_config(args, save=False)   # print model configuration of evaluation
    
    # attach timer function
    if args.timer:
        preprocessor = timer(preprocessor)

        # differnet demo env
        demo_test = timer(demo_test)
        demo_eval = timer(demo_eval)

        postprocessor = timer(postprocessor)
        verification = timer(verification)
        copy2usb = timer(copy2usb)
        translation = timer(translation)

    # execution main function
    demo_func = efficient_demo if not args.nonefficient else demo
    if args.gui:
        demo_func(args,noise,word_idx)
    else:
        demo_func(args)

    # timer statistics
    if args.timer:
        import pandas as pd
        stat = pd.DataFrame(exe_stat, index=['preprocessor', 'demo_efficient', 'postprocessor', 'verification', 'translation', 'copy2usb'], columns=['time'])

        stat['percent'] = stat / stat.sum() * 100

        stat = stat.round(
            {'time': 2, 'percent': 2,}
        )
        print(f'\nperformance statistics:\n{stat}')
        print(f'total execution time: {stat["time"].sum()}')

if __name__ == '__main__':
    # argument setting
    args = argument_setting()
    demo_main(args)