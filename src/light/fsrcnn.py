# Light
# TODO: Add new features in Light Modules, make it can execute normally
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class LightFSRCNN(pl.LightningModule):
    def __init__(self, args, scale_factor=1, num_channels=1, d=56, s=12, m=4, criterion=nn.MSELoss()):
        super(LightFSRCNN, self).__init__()

        self.args = args
        self.criterion = criterion
        self.model = FSRCNN(scale_factor, num_channels, d, s, m)

        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.eval()

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        self.train_set = AxisDataSet(self.args.train_path)
        self.test_set = AxisDataSet(self.args.test_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.args.lr)

    # training
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

    def training_step(self, data, idx):
        inputs, target = data
        outputs = self(inputs)
        
        # mse loss
        mse_loss = self.criterion(outputs, target)

        # content loss
        gen_features = self.feature_extractor(outputs)
        real_features = self.feature_extractor(target)
        content_loss = self.criterion(gen_features, real_features)

        # total loss
        loss = mse_loss + content_loss

        if self.logger is not None:
            self.logger.experiment.add_scalar('train_loss', loss)

        if self.current_epoch % self.args.check_interval:
            out2csv(inputs, f'{self.current_epoch}_input', self.args.stroke_length)
            out2csv(outputs, f'{self.current_epoch}_output', self.args.stroke_length)
            out2csv(target, f'{self.current_epoch}_target', self.args.stroke_length)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([
            x['loss'] for x in outputs
        ]).mean()

        logs = {'loss': loss_mean}
        return {
            'progress_bar': logs,
            'loss': loss_mean
        }

    # valid
    def val_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

    # test
    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

    def test_step(self, data, idx):
        inputs, target = data
        outputs = self(inputs)
        
        # mse loss
        mse_loss = self.criterion(outputs, target)

        # content loss
        gen_features = self.feature_extractor(outputs)
        real_features = self.feature_extractor(target)
        content_loss = self.criterion(gen_features, real_features)

        # total loss
        loss = mse_loss + content_loss
        if self.logger is not None:
            self.logger.experiment.add_scalar('test_loss', loss)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        loss_mean = torch.stack([
            x['test_loss'] for x in outputs
        ]).mean()

        logs = {'test_loss': loss_mean}
        return {
            'progress_bar': logs,
            'test_loss': loss_mean
        }