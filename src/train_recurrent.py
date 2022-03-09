import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
from train import train_argument
from model import FeatureExtractor
from utils import writer_builder, model_builder, out2csv, inverse_scaler_transform, save_final_predict_and_new_dataset, model_config
from dataset import AxisDataSet, cross_validation

# TODO: change path name, add other args
def train(model, train_loader, valid_loader, optimizer, criterion, args):
    # declare content loss
    best_err = None
    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.eval()

    # load data
    model_path = f'fsrcnn_{args.scale}x.pt'
    checkpoint = {'epoch': 1}   # start from 1

    # load model from exist .pt file
    if args.load is True and os.path.isfile(model_path):
        r"""
        load a pickle file from exist parameter

        state_dict: model's state dict
        epoch: parameters were updated in which epoch
        """
        checkpoint = torch.load(model_path, map_location=f'cuda:{args.gpu_id}')
        checkpoint['epoch'] += 1    # start from next epoch
        model.load_state_dict(checkpoint['state_dict'])

    # store the training time
    writer = writer_builder(args.log_path,args.model_name)

    for epoch in range(checkpoint['epoch'], args.epochs+1):
        model.train()
        err = 0.0
        valid_err = 0.0

        store_data_cnt = 0  # to create new dataset

        for data in tqdm(train_loader, desc=f'train epoch: {epoch}/{args.epochs}'):
            # read data from data loader
            inputs, target, stroke_num = data
            inputs, target = inputs.cuda(), target.cuda()

            # predicted fixed 6 axis data
            pred = model(inputs)

            # inverse transform
            pred = inverse_scaler_transform(pred, inputs)

            # MSE loss
            mse_loss = criterion(pred, target)

            # content loss
            gen_features = feature_extractor(pred)
            real_features = feature_extractor(target)
            content_loss = criterion(gen_features, real_features)

            # for compatible but bad for memory usage
            loss = mse_loss + content_loss

            err += loss.sum().item() * inputs.size(0)

            # out2csv every check interval epochs (default: 5)
            if epoch % args.check_interval == 0:
                out2csv(inputs, f'{epoch}_input', args.stroke_length)
                out2csv(pred, f'{epoch}_output', args.stroke_length)
                out2csv(target, f'{epoch}_target', args.stroke_length)

            if epoch  == args.epochs:
                if not os.path.exists('final_output'):
                    os.mkdir('final_output')
                save_final_predict_and_new_dataset(pred, stroke_num, f'final_output/', args, store_data_cnt)
                store_data_cnt+=args.batch_size
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # cross validation
        model.eval()
        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f'valid epoch: {epoch}/{args.epochs}'):

                inputs, target = data
                inputs, target = inputs.cuda(), target.cuda()

                pred = model(inputs)

                # inverse transform
                pred = inverse_scaler_transform(pred, inputs)

                # MSE loss
                mse_loss = criterion(pred, target)

                # content loss
                gen_features = feature_extractor(pred)
                real_features = feature_extractor(target)
                content_loss = criterion(gen_features, real_features)

                # for compatible
                loss = mse_loss + content_loss

                valid_err += loss.sum().item() * inputs.size(0)

        if epoch  == args.epochs:
            save_final_predict_and_new_dataset(pred, stroke_num, f'final_output/', args, store_data_cnt)
            store_data_cnt+=args.batch_size

        # compute loss
        err /= len(train_loader.dataset)
        valid_err /= len(valid_loader.dataset)
        print(f'train loss: {err:.4f}, valid loss: {valid_err:.4f}')

        # update every epoch
        # save model as pickle file
        if best_err is None or err < best_err:
            best_err = err

            # save current epoch and model parameters
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                }
                , model_path)

        # update loggers
        writer.add_scalars('Loss/', {'train loss': err,
                                          'valid loss': valid_err}, epoch)

    writer.close()


if __name__ == '__main__':
    # argument setting
    train_args = train_argument()

    # config
    model_config(train_args, save=True)     # save model configuration before training

    # set cuda
    torch.cuda.set_device(train_args.gpu_id)

    # model
    model = model_builder(train_args.model_name, train_args.scale, *train_args.model_args).cuda()

    # optimizer and criteriohn
    optimizer = optim.Adam(model.parameters(), lr=train_args.lr)
    criterion = nn.MSELoss()

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

    # training
    train(model, train_loader, valid_loader, optimizer, criterion, train_args)

    # config
    model_config(train_args, save=False)     # print model configuration after training
