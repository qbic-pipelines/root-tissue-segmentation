Usage
=============

Setup
-------

mlf-core based mlflow projects require either Conda or Docker to be installed.
The usage of Docker is highly preferred, since it ensures that system-intelligence can fetch all required and accessible hardware.
This cannot be guaranteed for MacOS let alone Windows environments.

Conda
+++++++

There is no further setup required besides having Conda installed and CUDA configured for GPU support.
mlflow will create a new environment for every run.

Docker
++++++++

If you use Docker you should not need to build the Docker container manually, since it should be available on Github Packages or another registry.
However, if you want to build it manually for e.g. development purposes, ensure that the names matches the defined name in the ``MLproject``file.
This is sufficient to train on the CPU. If you want to train using the GPU you need to have the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-docker>`_ installed.

Training
-----------

Training on the CPU
+++++++++++++++++++++++

Set your desired environment in the MLproject file. Start training using ``mlflow run .``.
No further parameters are required.

Training using GPUs
+++++++++++++++++++++++

Regularly used commands: ``mlflow run . -A gpus=all``, ``mlflow run . -A gpus=all -P max_epochs=2``.

Please see the `mlflow documentation <https://www.mlflow.org/docs/latest/cli.html#mlflow-run>`_

Conda environments will automatically use the GPU if available.
Docker requires the accessible GPUs to be passed as runtime parameters. To train using all gpus run ``mlflow run . -A t-A gpus=all -P gpus=<<num_of_gpus>> -P acc=ddp``.
To train only on CPU it is sufficient to call ``mlflow run . -A t``. To train on a single GPU, you can call ``mlflow run . -A t -A gpus=all -P gpus=1`` and for multiple GPUs (for example 2)
``mlflow run . -A t -A gpus=all -P gpus=2 -P accelerator=ddp``.
You can replace ``all`` with specific GPU ids (e.g. 0) if desired.

Parameters
+++++++++++++++
- model:						Semanti segmentation model (U-Net, U-Net++, U2-Net)      ['u2net':	string]
- gpus:							Number of gpus to train with                             [2:	int]
- max_epochs:					Number of epochs to train                                [74:	int]
- lr:							Learning rate of the optimizer                           [0.0005739979509018617:	float]
- training-batch-size:			Batch size for training batches                          [10:	int]
- test-batch-size:				Batch size for test batches                              [60:	int]
- accelerator:					Accelerator connecting to the Lightning Trainer          ['dp':	string]
- gamma-factor:					Gamma for the Focal Loss function                        [2.8074708243593878:	float]
- weight-decay:					Weight decay for the AdaBelief optimizer                 [0.08843850663784153:	float]
- epsilon:						Epsilon for the AdaBelief optimizer                      [2.1335101419972938e-14:	float]
- alpha-0:						Alpha for Focal Loss function                            [0.39228916176765655:	float]
- alpha-1:						Alpha for Focal Loss function                            [0.4476337434213018:	float]
- alpha-2:						Alpha for Focal Loss function                            [0.8902483094365905:	float]
- alpha-3:						Alpha for Focal Loss function                            [0.6695418278351652:	float]
- alpha-4:						Alpha for Focal Loss function                            [0.9382751049017035:	float]
- log-interval:					Number of batches to train for before logging            [100:	int]
- general-seed:					Python, Random, Numpy seed                               [0:	int]
- pytorch-seed:					Pytorch specific seed                                    [0:	int]
