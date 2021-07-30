import pytorch_lightning as pl
from albumentations import Compose
from albumentations.augmentations import transforms
from torch.utils.data import DataLoader

from data_loading.phdfm import PHDFM


class PHDFMDataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        """
        Initialization of the data module with a PHDFM and test dataset, as well as a loader for each.
        The dataset is the PHDFM dataset.
        """
        super(PHDFMDataModule, self).__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs
        # transforms for images, picked after evaluation of experimental environment.
        self.transform = Compose(
            [transforms.Normalize(0.6993, 0.4158, 255, always_apply=True),
             transforms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, always_apply=False,
                                         p=0.5)],
        )

        self.setup()
        self.prepare_data()

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into PHDFM, test, validation data
        :param stage: Stage - training or testing
        """
        self.df_train = PHDFM(root='rsphd/dataset', set="training", transform=self.transform, download=True)
        # Val and Test are currently the same, as the final application is the "test" set.
        self.df_val = PHDFM('rsphd/dataset', set="validation",
                            transform=Compose([transforms.Normalize(0.6993, 0.4158, 255, always_apply=True)]),
                            download=True)
        self.df_test = PHDFM('rsphd/dataset', set="test",
                             transform=Compose([transforms.Normalize(0.6993, 0.4158, 255, always_apply=True)]),
                             download=True)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(self.df_train, batch_size=self.args['training_batch_size'],
                          num_workers=self.args['num_workers'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(self.df_val, batch_size=self.args['test_batch_size'],
                          num_workers=self.args['num_workers'], shuffle=False, pin_memory=True)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return DataLoader(self.df_test, batch_size=self.args['test_batch_size'], num_workers=self.args['num_workers'],
                          shuffle=False, pin_memory=True)

    def prepare_data(self, *args, **kwargs):
        pass
