#TODO: add amp training
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
from model import FeatureExtractor
from utils import (StorePair, writer_builder, model_builder, optimizer_builder, criterion_builder, schedule_builder,
                out2csv, NormScaler, model_config, summary, config_loader, EarlyStopping)
from dataset import AxisDataSet, cross_validation
from postprocessing import postprocessor

def train_argument(inhert=False):
    """return train arguments

    Args:
        inhert (bool, optional): return parser for compatiable. Defaults to False.

    Returns:
        parser_args(): if inhert is false, return parser's arguments
        parser(): if inhert is true, then return parser
    """

    # for compatible
    parser = ArgumentParser(add_help=not inhert)

    # doc setting
    parser.add_argument('--doc', type=str, metavar='./doc/sampleV4.yaml',
                        help='load document file by position(default: None)')

    # dataset setting
    parser.add_argument('--stroke-length', type=int, default=150,
                        help='control the stroke length (default: 150)')
    parser.add_argument('--train-path', type=str, default='/home/jefflin/dataset/train',
                        help='training dataset path (default: /home/jefflin/dataset/train)')
    parser.add_argument('--target-path', type=str, default='/home/jefflin/dataset/target',
                        help='target dataset path (default: /home/jefflin/dataset/target)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='set the batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='set the number of processes to run (default: 8)')
    parser.add_argument('--holdout-p', type=float, default=0.8,
                        help='set hold out CV probability (default: 0.8)')
    parser.add_argument('--mean', type=float, default=0.5,
                        help='set mean value to normalize data (default: 0.5)')
    parser.add_argument('--std', type=float, default=0.5,
                        help='set std value to normalize data (default: 0.5)')

    # model setting
    parser.add_argument('--model-name', type=str, default='FSRCNN',
                        metavar='FSRCNN, DDBPN' ,help="set model name (default: 'FSRCNN')")
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')
    parser.add_argument('--model-args', action=StorePair, nargs='+', default={},
                        metavar='key=value', help="set other args (default: {})")
    parser.add_argument('--optim', type=str, default='rangerVA',
                        help='set optimizer')
    parser.add_argument('--load', action='store_true', default=False,
                        help='load model parameter from exist .pt file (default: False)')
    parser.add_argument('--version', type=int, dest='load',
                        help='load specific version (default: False)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='set the learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0,
                        help="set weight decay (default: 0)")

    # training setting
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help="set loss 1's weight (default: 1e-3)")
    parser.add_argument('--beta', type=float, default=1,
                        help="set loss 2's weight (default: 1)")
    parser.add_argument('--epochs', type=int, default=50,
                        help='set the epochs (default: 50)')
    parser.add_argument('--check-interval', type=int, default=5,
                        help='setting output a csv file every epoch of interval (default: 5)')
    parser.add_argument('--criterion', type=str, default='huber',
                        help="set criterion (default: 'huber')")
    parser.add_argument('--amp', action='store_true', default=False,
                        help='training with amp (default: False)')

    # Sceduler setting
    parser.add_argument('--scheduler', action='store_true', default=False,
                        help='training with step or multi step scheduler (default: False)')
    parser.add_argument('--lr-method', type=str, dest='scheduler', 
                        help='training with chose lr scheduler (default: False)')
    parser.add_argument('--step', nargs='+', default=2,
                        help='decreate learning rate every few epochs (default: 2)')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='set decreate factor (default: 0.1)')

    # Early-Stop setting
    parser.add_argument('--early-stop', action='store_false', default=True,
                        help='Early stops the training if validation loss does not improve (default: True)')
    parser.add_argument('--patience', type=int, default=5,
                        help='How long to wait after last time validation loss improved. (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Minimum change in the monitored quantity to qualify as an improvement. (default: 0.001)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If True, prints a message for each validation loss improvement. (default: False)')

    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='set the logger path of pytorch model (default: ./logs)')
    
    # save setting
    parser.add_argument('--save-path', type=str, default='./output',
                        help='set the output file (csv or txt) path (default: ./output)')
    parser.add_argument('--out-num', type=int, default=5,
                        help='Set the number of model porcess data to get in one epoch. (default: 5)')

    # for the compatiable
    if inhert is True:
        return parser
    
    return parser.parse_args()


def train(model, train_loader, valid_loader, optimizer, criterion, args):
    # content_loss
    best_err = None
    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.eval()

    writer, log_path = writer_builder(
        args.log_path, args.model_name, load=args.load)
    
    # init data
    checkpoint = {
        'epoch': 1,         # start from 1
        'train_iter': 0,    # train iteration
        'valid_iter': 0,    # valid iteration
    }   
    model_path = os.path.join(log_path, f'{args.model_name}_{args.scale}x.pt')

    # config
    model_config(train_args, save=log_path)     # save model configuration before training

    # load model from exist .pt file
    if args.load and os.path.isfile(model_path):
        r"""
        load a pickle file from exist parameter

        state_dict: model's state dict
        epoch: parameters were updated in which epoch
        """
        checkpoint = torch.load(model_path, map_location=f'cuda:{args.gpu_id}')
        checkpoint['epoch'] += 1        # start from next epoch
        checkpoint['train_iter'] += 1
        checkpoint['valid_iter'] += 1
        model.load_state_dict(checkpoint['state_dict'])

    # initialize the early_stopping object
    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.patience, threshold=args.threshold, verbose=args.verbose, path=model_path)

    if args.scheduler:
        scheduler = schedule_builder(optimizer, args.scheduler, args.step, args.factor)

    # progress bar postfix value
    pbar_postfix = {
        'MSE loss': 0.0,
        'Content loss': 0.0,
        'lr': args.lr,
    }

    for epoch in range(checkpoint['epoch'], args.epochs+1):
        model.train()
        err = 0.0
        valid_err = 0.0

        train_bar = tqdm(train_loader, desc=f'Train epoch: {epoch}/{args.epochs}')
        for data in train_bar:
            # load data from data loader
            inputs, target, _ = data
            inputs, target = inputs.cuda(), target.cuda()

            # predicted fixed 6 axis data
            pred = model(inputs)

            # MSE loss
            mse_loss = args.alpha *criterion(pred - inputs, target - inputs)

            # content loss
            gen_features = feature_extractor(pred)
            real_features = feature_extractor(target)
            content_loss = args.beta * criterion(gen_features, real_features)

            # for compatible but bad for memory usage
            loss = mse_loss + content_loss

            # update progress bar
            pbar_postfix['MSE loss'] = mse_loss.item()
            pbar_postfix['Content loss'] = content_loss.item()

            # show current lr
            if args.scheduler:
                pbar_postfix['lr'] = optimizer.param_groups[0]['lr']

            train_bar.set_postfix(pbar_postfix)

            err += loss.sum().item() * inputs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update writer
            writer.add_scalar('Iteration/train loss', loss.sum().item(), checkpoint['train_iter'])
            checkpoint['train_iter'] += 1

        # cross validation
        valid_bar = tqdm(valid_loader, desc=f'Valid epoch:{epoch}/{args.epochs}', leave=False)
        model.eval()
        input_epoch = pred_epoch = target_epoch = torch.empty(0,0)
        with torch.no_grad():
            for data in valid_bar:
            # for data in valid_loader:
                inputs, target, _ = data
                inputs, target = inputs.cuda(), target.cuda()

                pred = model(inputs)

                # MSE loss
                mse_loss = criterion(pred - inputs, target - inputs)

                # content loss
                gen_features = feature_extractor(pred)
                real_features = feature_extractor(target)
                content_loss = criterion(gen_features, real_features)

                # for compatible
                loss = mse_loss + content_loss

                # update progress bar
                pbar_postfix['MSE loss'] = mse_loss.item()
                pbar_postfix['Content loss'] = content_loss.item()

                # show current lr
                if args.scheduler:
                    pbar_postfix['lr'] = optimizer.param_groups[0]['lr']

                valid_bar.set_postfix(pbar_postfix)

                valid_err += loss.sum().item() * inputs.size(0)

                # update writer
                writer.add_scalar('Iteration/valid loss', loss.sum().item(), checkpoint['valid_iter'])
                checkpoint['valid_iter'] += 1

                # out2csv every check interval epochs (default: 5)
                if epoch % args.check_interval == 0:
                    input_epoch = inputs
                    pred_epoch = pred
                    target_epoch = target

        # out2csv every check interval epochs (default: 5)
        if epoch % args.check_interval == 0:

            # tensor to csv file
            out2csv(input_epoch, f'{epoch}', 'input', args.out_num, args.save_path, args.stroke_length)
            out2csv(pred_epoch, f'{epoch}', 'output', args.out_num, args.save_path, args.stroke_length)
            out2csv(target_epoch, f'{epoch}', 'target', args.out_num, args.save_path, args.stroke_length)

        # compute loss
        err /= len(train_loader.dataset)
        valid_err /= len(valid_loader.dataset)
        print(f'\ntrain loss: {err:.4f}, valid loss: {valid_err:.4f}')

        # update scheduler
        if args.scheduler:
            scheduler.step()

        # update loggers
        writer.add_scalars('Epoch',
                           {'train loss': err, 'valid loss': valid_err},
                           epoch,)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if args.early_stop:
            early_stopping(valid_err, model, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        # if early stop is false, store model when the err is lowest
        elif epoch == checkpoint['epoch'] or err < best_err:
            best_err = err  # save err in first epoch

            # save current epoch and model parameters
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'train_iter': checkpoint['train_iter'],
                    'valid_iter': checkpoint['valid_iter'],
                }
                , model_path)

    writer.close()


if __name__ == '__main__':
    # argument setting
    train_args = train_argument()

    # replace args by document file
    if train_args.doc:
        train_args = config_loader(train_args.doc, train_args)
    # set cuda
    torch.cuda.set_device(train_args.gpu_id)

    # model
    model = model_builder(train_args.model_name, train_args.scale, **train_args.model_args).cuda()
    
    # optimizer and critera
    optimizer = optimizer_builder(train_args.optim) # optimizer class
    optimizer = optimizer(                          # optmizer instance
        model.parameters(),
        lr=train_args.lr,
        weight_decay=train_args.weight_decay
    )
    criterion = criterion_builder(train_args.criterion)

    # dataset
    full_set = AxisDataSet(train_args.train_path, train_args.target_path)

    # build hold out CV
    train_set, valid_set = cross_validation(
        full_set,
        mode='hold',
        p=train_args.holdout_p,)

    # dataloader
    train_loader = DataLoader(train_set,
                              batch_size=train_args.batch_size,
                              shuffle=True,
                              num_workers=train_args.num_workers,
                            #   sampler=train_sampler,
                              pin_memory=False,)
    valid_loader = DataLoader(valid_set,
                              batch_size=train_args.batch_size,
                              shuffle=False,
                              num_workers=train_args.num_workers,
                            #   sampler=valid_sampler,
                              pin_memory=False,)

    # model summary
    data, _, _ = train_set[0]
    summary(model,
        tuple(data.shape),
        batch_size=train_args.batch_size,
        device='cuda',
        model_name=train_args.model_name.upper(),
        )
    
    # training
    train(model, train_loader, valid_loader, optimizer, criterion, train_args)

    # config
    model_config(train_args, save=False)     # print model configuration after training

    postprocessor(train_args.save_path)