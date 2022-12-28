import os
from argparse import ArgumentParser

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print

from data_loading.data_loader import PHDFMDataModule
from mlf_core.mlf_core import MLFCore
from models import unet
from models.unet import U2NET
from random import randint
if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch Autolog PHDFM Example')
    parser.add_argument(
        '--general-seed',
        type=int,
        default=0,
        help='General random seed',
    )
    parser.add_argument(
        '--pytorch-seed',
        type=int,
        default=0,
        help='Random seed of all Pytorch functions',
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='log interval of stdout',
    )
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = U2NET.add_model_specific_args(parent_parser=parser)
    mlflow.pytorch.autolog(log_models=False)
    # log conda env and system information
    MLFCore.log_sys_intel_conda_env()
    # parse cli arguments
    args = parser.parse_args()
    dict_args = vars(args)
    # store seed
    # number of gpus to make linter bit less restrict in terms of naming
    general_seed = dict_args['general_seed']
    pytorch_seed = dict_args['pytorch_seed']
    num_of_gpus = dict_args['gpus']
    MLFCore.set_general_random_seeds(general_seed)
    MLFCore.set_pytorch_random_seeds(pytorch_seed, num_of_gpus)

    if 'accelerator' in dict_args:
        if dict_args['accelerator'] == 'None':
            dict_args['accelerator'] = None
        elif dict_args['accelerator'] != 'dp':
            print(
                f'[bold red]{dict_args["accelerator"]}[bold blue] currently not supported. Switching to [bold green]ddp!')
            dict_args['accelerator'] = 'ddp'
    dm = PHDFMDataModule(**dict_args)

    MLFCore.log_input_data('root_tissue_segmentation/dataset/PHDFM')
    if 'class_weights' not in dict_args.keys():
        weights = dm.df_train.class_weights
        wt_list = weights.query('set_name=="training"')['weights'].tolist()
        dict_args['class_weights'] = wt_list

    dm.setup(stage='fit')
    # Supported batch size:24
    # Supported batch size:96
    model = getattr(unet, dict_args['model'].upper())
    model = model(5, len_test_set=len(dm.df_test), hparams=parser.parse_args(), input_channels=1, min_filter=64,
                  **dict_args)
    model.log_every_n_steps = dict_args['log_interval']

    # check, whether the run is inside a Docker container or not
    if 'MLF_CORE_DOCKER_RUN' in os.environ:
        checkpoint_callback = ModelCheckpoint(dirpath='/data/mlruns', save_top_k=1, verbose=True,
                                              monitor='val_avg_iou', mode='max', filename='best')
        trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback, default_root_dir='/data',
                                                logger=TensorBoardLogger('/data'))
        tensorboard_output_path = f'data/default/version_{trainer.logger.version}'
    else:
        checkpoint_callback = ModelCheckpoint(dirpath='/data/mlruns', save_top_k=1,
                                              verbose=True, monitor='val_avg_iou', mode='max', filename=f'best{randint(0, 10000)}')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor],
                                                default_root_dir=os.getcwd() + "/mlruns",
                                                logger=TensorBoardLogger('/data'), auto_lr_find=False)
        tensorboard_output_path = f'data/default/version_{trainer.logger.version}'

    trainer.deterministic = True
    trainer.benchmark = False
    # lrfind = trainer.tuner.lr_find(model,dm)
    # print(lrfind.suggestion())
    trainer.fit(model, dm)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path,datamodule=dm)
    #trainer.save_checkpoint("/data/example.ckpt")
    print(f'\n[bold blue]For tensorboard log, call [bold green]tensorboard --logdir={tensorboard_output_path}')
    print(checkpoint_callback.best_model_score.item())
    with open('/data/best.txt', 'w') as f:
        f.write(str(checkpoint_callback.best_model_score.item()))
        
