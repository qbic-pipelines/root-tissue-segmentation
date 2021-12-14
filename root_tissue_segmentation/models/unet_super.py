import abc
from argparse import ArgumentParser
from typing import Any, Optional

import cv2
import pytorch_lightning as pl
import torch
from pytorch_toolbelt.losses import FocalLoss
from torch_optimizer import AdaBelief

__all__ = ['UNetsuper']

from metric.iou import iou_score
from utils import label2rgb, unnormalize


class UNetsuper(pl.LightningModule):
    def __init__(self, num_classes, len_test_set: int, hparams: dict, input_channels=1, min_filter=32, **kwargs):
        super(UNetsuper, self).__init__()
        self.num_classes = num_classes
        self.metric = iou_score
        self.save_hyperparameters(hparams)
        self.args = kwargs
        self.len_test_set = len_test_set
        self.weights = kwargs['class_weights']
        self.alphas = [kwargs[f'alpha_{x}'] for x in range(5)]
        self.criterion = FocalLoss(alpha=kwargs['alpha_1'], gamma=kwargs['gamma_factor'])
        """
        self.criterion = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.tensor(self.weights),
            gamma=2,
            reduction='mean',
            force_reload=False
        )
        """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers (default: 3)')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
        parser.add_argument('--gamma-factor', type=float, default=2.0, help='learning rate (default: 0.01)')
        parser.add_argument('--weight-decay', type=float, default=1e-5, help='learning rate (default: 0.01)')
        parser.add_argument('--epsilon', type=float, default=1e-16, help='learning rate (default: 0.01)')
        parser.add_argument('--alpha-0', type=float, default=1, help='learning rate (default: 0.01)')
        parser.add_argument('--alpha-1', type=float, default=1, help='learning rate (default: 0.01)')
        parser.add_argument('--alpha-2', type=float, default=1, help='learning rate (default: 0.01)')
        parser.add_argument('--alpha-3', type=float, default=1, help='learning rate (default: 0.01)')
        parser.add_argument('--alpha-4', type=float, default=1, help='learning rate (default: 0.01)')
        parser.add_argument('--model', type=str, default="u2net", help='learning rate (default: 0.01)')
        parser.add_argument('--training-batch-size', type=int, default=20, help='Input batch size for training')
        parser.add_argument('--test-batch-size', type=int, default=1000, help='Input batch size for testing')
        return parser

    @abc.abstractmethod
    def forward(self, x):
        pass

    def loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        labels = labels.long()
        return self.criterion(logits.squeeze(), labels.squeeze())

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        x, y, logits, y_hat = self.predict(train_batch, batch_idx)
        loss = self.loss(logits, y)
        self.train_iou = self.metric(y_hat, y)
        self.log('train_IoU', self.train_iou[0].mean(), on_step=False, on_epoch=True)
        for i in range(self.num_classes):
            self.log(f'train_IoU_{i}', self.train_iou[0][i], on_step=False, on_epoch=True)
        return {'loss': loss, 'iou': self.train_iou[0].mean()}

    def training_epoch_end(self, training_step_outputs):
        """
        On each training epoch end, log the average training loss
        """
        train_avg_loss = torch.stack([train_output['loss'] for train_output in training_step_outputs]).mean()
        self.log('train_avg_loss', train_avg_loss, sync_dist=True)
        train_avg_iou = torch.stack([train_output['iou'] for train_output in training_step_outputs]).mean()
        self.log('train_avg_iou', train_avg_iou, sync_dist=True)

    def validation_step(self, val_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        data, target, output, prediction = self.predict(val_batch, batch_idx)
        self.val_iou = self.metric(prediction, target)
        loss = self.loss(output, target)
        self.log('val_IoU', self.val_iou[0].mean(), on_step=True, on_epoch=True, sync_dist=True)
        self.log_tb_images(batch_idx, data, target, prediction)
        for i in range(self.num_classes):
            self.log(f'val_IoU_{i}', self.val_iou[0][i], on_step=True, on_epoch=True, sync_dist=True)
        return {'loss': loss,
                'iou': torch.mean(torch.stack([self.val_iou[0][2], self.val_iou[0][3], self.val_iou[0][4]]))}

    def validation_epoch_end(self, validation_step_outputs):
        """
        On each training epoch end, log the average training loss
        """
        val_avg_loss = torch.stack([val_output['loss'] for val_output in validation_step_outputs]).mean()
        self.log('val_avg_loss', val_avg_loss, sync_dist=True)
        val_avg_iou = torch.stack([val_output['iou'] for val_output in validation_step_outputs]).mean()
        self.log('val_avg_iou', val_avg_iou, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        Predicts on the test dataset to compute the current accuracy of the models.

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """

        data, target, output, prediction = self.predict(test_batch, batch_idx)
        self.test_iou = self.metric(prediction, target)
        self.log('test_IoU', self.test_iou[0].mean(), on_step=False, on_epoch=True, sync_dist=True)
        for i in range(self.num_classes):
            self.log(f'test_IoU_{i}', self.test_iou[0][i], on_step=False, on_epoch=True, sync_dist=True)
        # sum up batch loss
        test_loss = self.loss(output, target)
        # get the index of the max log-probability
        correct = prediction.eq(target.data).sum()
        return {'test_loss': test_loss, 'correct': correct}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score

        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """
        avg_test_loss = sum([test_output['test_loss'] for test_output in outputs]) / self.len_test_set
        test_correct = float(sum([test_output['correct'] for test_output in outputs]))
        self.log('avg_test_loss', avg_test_loss, sync_dist=True)
        self.log('test_correct', test_correct, sync_dist=True)

    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        data, target = batch
        output = self.forward(data)
        _, prediction = torch.max(output, dim=1)
        return data, target, output, prediction

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = AdaBelief(self.parameters(), lr=self.hparams['lr'], eps=self.args['epsilon'],
                                   betas=(0.9, 0.999),
                                   weight_decay=self.args['weight_decay'],
                                   weight_decouple=True)
        print(self.hparams["lr"])
        self.scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.hparams['lr'],
                                                             total_steps=self.args["max_epochs"],
                                                             pct_start=0.45, three_phase=True),
            'monitor': 'train_avg_loss',
        }
        return [self.optimizer], [self.scheduler]

    def log_tb_images(self, batch_idx, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, index=0):
        img = cv2.cvtColor(unnormalize(x[index].cpu().detach().numpy().squeeze()), cv2.COLOR_GRAY2RGB).astype(int)
        pred = y_hat[index].cpu().detach().numpy()
        mask = y[index].cpu().detach().numpy()
        alpha = 0.7
        # Performing image overlay
        gt = label2rgb(alpha, img, mask)
        prediction = label2rgb(alpha, img, pred)
        log = torch.stack([gt, prediction], dim=0)
        self.logger.experiment.add_images(f'Images and Masks Batch: {batch_idx}', log, self.current_epoch)
