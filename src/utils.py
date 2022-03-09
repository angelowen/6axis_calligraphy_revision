import re
import os
import csv
import sys
import json
import shutil
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import Action, ArgumentParser, Namespace
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Union, Optional


def _key_func(key):
    """
    manipulate key to sort list by numerical suffix

    Args:
        key (str): element of list

    Returns:
        [int]: key which is int 
    """

    return int(key.split('_')[-1])

################################################################
########################## model info ##########################
################################################################


class StorePair(Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(StorePair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        target = {}

        for pair in values:
            k, v = pair.split('=')

            if re.match(r'^-?\d+\.\d+$', v):
                v = float(v)
            elif v.isdigit():
                v = int(v)

            target[k] = v
        
        # assign value to namespace
        setattr(namespace, self.dest, target)


def writer_builder(log_root, model_name, load: Union[bool, int]=False):
    """Build writer acording to exist or new logs

    save summary writer in: ./log_root/model_name/version_*

    Args:
        log_root (str): logs root
        model_name (str): model's name
        load (Union[bool, int], optional): load existed Tensorboard. Defaults to False.

    Returns:
        SummaryWriter: tensorboard
        log_path(str): determine version location
    """

    from torch.utils.tensorboard import SummaryWriter

    if not os.path.exists(log_root):
        os.mkdir(log_root)

    log_root = os.path.join(log_root, model_name.upper())
    print('\n####### logger info #######\n')

    # make sure logs directories exist
    if not os.path.exists('./logs'):
        os.mkdir('logs')

    if not os.path.exists(log_root):
        os.mkdir(log_root)

    # list version of model
    version = os.listdir(log_root)
    version.sort(key=_key_func)

    # load exist logs
    if version and type(load) is int:
        # check log path is exist or not
        if f'version_{load}' not in version:
            print(f'Logger Error: load non existent writer: {log_path}')
            print('\n####### logger info #######\n')
            os._exit(0)

        log_path = os.path.join(log_root, f'version_{load}')
        # print(f'load exist logger:{log_path}')

    # load specific version
    elif version and load is True:
        log_path = os.path.join(log_root, version[-1])
        # print(f'load exist logger:{log_path}')

    # create new log directory indexed by exist directories
    else:
        log_path = os.path.join(log_root, f'version_{len(version)}')
        os.mkdir(
            log_path
        )
        print(f'create new logger in:{log_path}')
    
    print(f'Tensorboard logger save in:{log_path}')
    print('\n####### logger info #######\n')

    return SummaryWriter(log_path), log_path


def model_builder(model_name, *args, **kwargs):
    """choose which model would be training

    Args:
        model_name (str): FSRCNN, DDBPN, DBPN, ADBPN

    model args:
    FSRCNN: scale_factor, num_channels=1, d=56, s=12, m=4
    DDBPN:  scale_factor, num_channels=1, stages=7, n0=256, nr=64
    DBPN:   scale_factor, num_channels=1, stages=7, n0=256, nr=64
    ADBPN:  scale_factor, num_channels=1, stages=7, n0=256, nr=64, col_slice=3, stroke_len=150

    Returns:
        model(torch.nn.module): instantiate model
    """
    from model import FSRCNN, DDBPN, DBPN, ADBPN

    # class object, yet instantiate
    model = {
        'fsrcnn': FSRCNN,    # scale_factor, num_channels=1, d=56, s=12, m=4
        'ddbpn': DDBPN,      # scale_factor, num_channels=1, stages=7, n0=256, nr=64
        'dbpn': DBPN,        # scale_factor, num_channels=1, stages=7, n0=256, nr=64
        'adbpn': ADBPN,      # scale_factor, num_channels=1, stages=7, n0=256, nr=64, col_slice=3, stroke_len=150
        'lapsrn': NotImplementedError,
        'drln': NotImplementedError,
    }.get(model_name.lower())

    return model(*args, **kwargs)


def model_config(args, save: Union[str, bool]=False):
    """record model configuration

    save model config as config.json
        if save is path, save to the path
        if save is True, save in current directory

    Args:
        args (Argparse object): Model setting
        save (Union[str, bool], optional): save as json file or just print to stdout. Defaults to False.
    """
    print('\n####### model arguments #######\n')
    for key, value in vars(args).items():
        
        # format modified
        value = {
            'model_name': f'{value}'.upper(),
        }.get(key, value)

        print(f'{key}: {value}')
    print('\n####### model arguments #######\n')

    if save:
    
        # if user has determined path or not
        save_path = save if type(save) is str else './'

        # if user specify doc
        if args.doc:
            shutil.copy(
                args.doc, 
                Path(save_path).joinpath(Path(args.doc).name)
            )
        
        # save config as .json file
        else:
            config_path = os.path.join(save_path, 'config.json')

            with open(config_path, 'w') as config:
                json.dump(vars(args), config, indent=4)


def config_loader(doc_path, args):
    """
    load config instead of argparser
    missing value will be filled by argparser's value

    Noticed that keys must be the same as original arguments
    support config type:
        .json
        .yaml (load with save loader)

    Args:
        doc_path (str): document path
        args : To be replaced arguments 

    Returns:
        argparse's object: for the compatiable 
    """
    with open(doc_path, 'r') as doc:
        format = doc_path.split('.')[-1]
        # determine the doc format
        load_func={
            'yaml': yaml.safe_load,
            'json': json.load,
        }[format]
        doc_args = load_func(doc)
    
    # remain doc in args
    """ try:
        del args.doc
        del doc_args['doc']
    except:
        print('There is no "doc" in args parser\n') """
        
    # remove test_path in train.py
    try:
        # train path exist
        if args.train_path:
            args.test_path = None

    except:
        print(f'No "train_path" founded in {sys.argv[0]}\n')

    # set load to True in test.py
    try:
        # test path exist
        if args.test_path and args.load is False:
            print('set --load to True')
            args.load = True
    except:
        print(f'No "test_path" founded in {sys.argv[0]}\n')
    
    # check which key value is missing
    arg_dict = vars(args)
    if arg_dict.keys() != doc_args.keys():
        for key in arg_dict.keys():
            if not key in doc_args.keys():
                print(f'"{key}" not found in document file!')

                # fill missing argument
                doc_args[key] = arg_dict[key]     
        
        print('\nWarning: missing above key in document file, which would raising error')
        print('Missing value would be argv value instead')
        # os._exit(0)


    print(f'config loaded: {doc_path}')
    return Namespace(**doc_args)


def optimizer_builder(optim_name: str):
    """build optimizer

    Args:
        optim_name (str): choose which optimizer for training
            'adam': optim.Adam
            'sgd': optim.SGD
            'ranger': Ranger
            'rangerva': RangerVA

    Returns:
        optimizer class, yet instantiate
    """
    from model import Ranger, RangerVA
    
    return {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'ranger': Ranger,   # Bug in Ranger
        'rangerva': RangerVA,
    }.get(optim_name.lower(), 'Error optimizer')


def summary(model, input_size, batch_size=-1, device="cuda", model_name: Optional[str]=None):
    """reference: https://github.com/sksq96/pytorch-summary
    
    modified to desired format

    Args:
        model (nn.module): torch model
        input_size (tuple, list): compute info
        batch_size (int, optional): Defaults to -1.
        device (str, optional): Control tensor dtype. Defaults to "cuda".
        model_name (Optional[str], optional): set model name or use class name. Defaults to None.
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # print model name
    name = model_name if model_name else model.__class__.__name__
    print(f'{name} summary')
    
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")

################################################################
########################### training ###########################
################################################################


def criterion_builder(criterion='mse', **kwargs):
    """build specific criterion

    mse: MSELoss
    rmse: RMSELoss
    huber: SmoothL1Loss
    L1: L1Loss

    Args:
        criterion (str, optional): to instantiate loss function. Defaults to 'mse'.

    Returns:
        nn.Module: return loss function
    """
    from model import RMSELoss

    return {
        'l1': nn.L1Loss(**kwargs),
        'mse': nn.MSELoss(**kwargs),
        'huber': nn.SmoothL1Loss(**kwargs),
        'rmse': RMSELoss(**kwargs),
    }[criterion.lower()]


def schedule_builder(optimizer, lr_method='step', step=2, gamma=0.1):
    """declare scheduler
    TODO: add lr scheduler by condition and fit in current function

    Args:
        optimizer : parameter of lr_scheduler 
        lr_method (str, optional): choose which scheduler to be used. Defaults to 'step'.
            step: step or multiple step
        step (int, optional): set step size. Defaults to 2 (epoch).
        gamma (float, optional): decrease factor. Defaults to 0.1.

    Returns:
        torch.optmizer.lr_sceduler
    """

    if lr_method == 'step' or lr_method is True:
        if type(step) is list and len(step) > 1:
            step = [int(x) for x in step]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, step, gamma)
        else:
            step = step.pop() if type(step) is list else step
            scheduler = optim.lr_scheduler.StepLR(optimizer, step, gamma)

    return scheduler


class NormScaler:
    """
    Normalize tensor's value into range 1~0
    And interse tensor back to original rage
    """
    def __init__(self, mean=0, std=1):
        self.min = None
        self.interval = None

        # normalize to specific range
        self.mean = mean
        self.std = std
    
    def fit(self, tensor):
        """transform tensor into range 1~0

        Args:
            tensor (torch.Tensor): unnormalized value

        Returns:
            shape as origin: inverse value
        """
        shape = tensor.shape
        tensor = tensor.view(shape[0], -1)

        self.min = tensor.min(1, keepdim=True)[0]
        self.interval = tensor.max(1, keepdim=True)[0] - self.min
        tensor = (tensor - self.min) / self.interval
        tensor = tensor.view(shape)

        # normailze
        if self.mean != 0 or self.std != 1:
            tensor.sub_(self.mean).div_(self.std)
        
        return tensor
        
    def inverse_transform(self, tensor):
        """inverse tensor's value back

        Args:
            tensor (torch.Tensor): normalized value

        Returns:
            shape as origin: inverse value 
        """
        assert self.min is not None, r'ValueError: scaler must fit data before inverse transform'
        assert self.interval is not None, r'ValueError: scaler must fit data before inverse transform'

        shape = tensor.shape

        # denormalize
        if self.mean != 0 or self.std != 1:
            tensor.mul_(self.std).add_(self.mean)

        tensor = tensor.view(shape[0], -1)

        tensor = tensor * self.interval + self.min

        return tensor.view(shape)


def inverse_scaler_transform(pred, target):
    """Inverse pred from range (0, 1) to target range.
    
    pred_inverse = (pred * (max - min)) + min
    
    ---
    Arguments:
        pred {torch.tensor} -- Tensor which is inversed from range (0, 1) to target range.
        target {torch.tensor} -- Inversion reference range.
    ---
    Returns:
        torch.tensor -- pred after inversed.
    """

    # max and min shape is [batch_size, 1, 1, 6]
    max = torch.max(target, 2, keepdim = True)[0]
    min = torch.min(target, 2, keepdim = True)[0]
    
    # pred_inverse = (pred * (max - min)) + min
    pred_inverse = torch.add(torch.mul(pred, torch.sub(max, min)), min)

    return pred_inverse


def out2csv(inputs, epoch, file_string, out_num, save_path, stroke_length, spec_flag = False):
    """
    store input to csv file.

    inputs: tensor data, with cuda device and size = [batch 1 STROKE_LENGTH 6]
    epoch: string, the epoch number
    file_string: string, 'input', 'output' or 'target'
    out_num: int, the number of model process data to get in one epoch
    save_path: string, the save path
    stroke_length: int, the length of each stroke 
    spec_flag: boolean, decide get specify number or not (default: False)

    no output
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    output = np.squeeze(inputs.cpu().detach().numpy(), axis=1)

    if spec_flag == False:
        table = output[0:out_num]
        for index in range(out_num):
            with open(f'{save_path}/{epoch}_{index}_{file_string}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(stroke_length):
                    row = [] * 7
                    row[1:6] = table[index][i][:]
                    row.append('stroke' + str(1))
                    writer.writerow(row)

    elif spec_flag == True:
        table = output[out_num]
        with open(f'{save_path}/{epoch}_{file_string}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(stroke_length):
                row = [] * 7
                row[1:6] = table[i][:]
                row.append('stroke' + str(1))
                writer.writerow(row)

def save_final_predict_and_new_dataset(inputs,stroke_num, file_string, args,store_data_cnt):
    output = np.squeeze(inputs.cpu().detach().numpy())
    
    for index in range(args.batch_size):
        try:
            table = output[index]
        except:
            break
        num = stroke_num[index]
        if not os.path.isdir(f'final_output/{num}'):
            # os.mkdir(f'new_train/{num}')
            os.mkdir(f'final_output/{num}')

        with open(f'{file_string}/{num}/{num}_{store_data_cnt+index}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(args.stroke_length):
                row = [] * 7
                row[1:6] = table[i][:]
                row.append(f'stroke{num}')
                writer.writerow(row)


#early stopping from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, threshold=0.001, verbose=False, path='./checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            threshold (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.001
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.threshold = threshold
        self.path = path

    def __call__(self, val_loss, model, epoch, checkpoint):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, checkpoint)

        # val_loss decrease less than 0.1% (default)
        elif score < self.best_score * (1. - self.threshold):
            self.counter += 1
            if self.verbose:
                print(f'{self.val_loss_min:.6f} --> {val_loss:.6f}: {100*(val_loss-self.val_loss_min)/self.val_loss_min:.2f}%')
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, checkpoint)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, checkpoint):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')

        # save current epoch and model parameters
        torch.save(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'train_iter': checkpoint['train_iter'],
                'valid_iter': checkpoint['valid_iter'],
            }
            , self.path)
        self.val_loss_min = val_loss
