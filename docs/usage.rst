Usage
=============

Setup
-------

Projects based on mlf-core require either Conda or Docker to be installed, we recommend to install both. The usage of Docker is highly preferred to run the codebase, since it ensures that system-intelligence can fetch all required and accessible hardware. This cannot be guaranteed for MacOS let alone Windows environments.

Conda
+++++++

It is required to have a Conda installed and CUDA configured for GPU support, mlflow will create a new environment for every run. Conda can be installed as instructed by the `Anaconda documentation <https://docs.anaconda.com/free/miniconda/>`_.

CUDA Toolkit
++++++++++++++

CUDA can be installed as instructed by the `CUDA documentation <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions>`_. Please note the `Post install steps <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions>`_.


Docker
++++++++

If you use Docker you should not need to build the Docker container manually, since it should be available on Github Packages or another registry. However, if you want to build it manually for e.g. development purposes, ensure that the names matches the defined name in the ``MLproject`` file. Docker can be installed as instructed by the `Docker documentation <https://docs.docker.com/engine/install/>`_.

This is sufficient to train on the CPU. If you want to train using the GPU you need to have the `NVIDIA Container Toolkit <https://github.com/NVIDIA/nvidia-container-toolkit>`_ installed.


Test Environment
++++++++++++++++++

This codebase has been tested in a virtual machine running **Ubuntu 22.04 LTS**. We installed Conda using the method suggested for Linux in the `Anaconda documentation <https://docs.anaconda.com/free/miniconda/#quick-command-line-install>`_. We installed **CUDA Toolkit 12.3 Update 2** for Ubuntu 22.04 using the instructions for the `network installation <https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network>`_. We installed Docker following the documentation for `Ubuntu <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`_. Please note the `Post install steps <https://docs.docker.com/engine/install/linux-postinstall/>`_. We installed Docker **Version 25.0.3, build 4debf41**. We followed the documentation to install the **NVIDIA Container Toolkit version 1.14.5** using `Apt <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt>`_.

We installed `Mlflow <https://mlflow.org/>`_ version **2.10.2** using the following conda command:

``conda install conda-forge::mlflow=2.10.2``

Training
-----------

Please see the `mlflow documentation <https://www.mlflow.org/docs/latest/cli.html#mlflow-run>`_. Set your desired environment in the ``MLproject`` file. A simple training test can be conducted by specifying a limited number of epochs, this can be done via a parameter in the mlflow command, e.g. ``mlflow run . --build-image -A runtime=nvidia -P max_epochs=3``.

Training on the CPU
+++++++++++++++++++++++

Training with CPU can be achived by specifying zero GPUs in the command, i.e. ``mlflow run . --build-image -A runtime=nvidia -P gpus=0``. It is useful to restrict the number of epochs, e.g. ``mlflow run . --build-image -A runtime=nvidia -P max_epochs=3 -P gpus=0``.

Training using GPUs
+++++++++++++++++++++++

Regularly used commands for development and testing:

- ``mlflow run . --build-image -A runtime=nvidia``
- ``mlflow run . --build-image -A runtime=nvidia -P max_epochs=3``
- ``mlflow run . --build-image -A runtime=nvidia -P max_epochs=3 -P gpus=1``
- ``mlflow run . --build-image -A runtime=nvidia -P max_epochs=3 -P gpus=2``

Conda environments will automatically use the GPU if available. Docker requires the accessible GPUs to be passed as runtime parameters. To train using all gpus run ``mlflow run . --build-image -A runtime=nvidia -P gpus=<<num_of_gpus>> -P acc=ddp``. 

To train on a single GPU, you can call ``mlflow run . --build-image -A runtime=nvidia -P gpus=1`` and for multiple GPUs (for example 2)
``mlflow run . --build-image -A runtime=nvidia -P gpus=2 -P accelerator=ddp``.

Hyperparameters and default values
-----------
- ``model``:				Semanti segmentation model (U-Net, U-Net++, U2-Net)      [``'u2net'``:	string]
- ``gpus``:					Number of gpus to train with                             [``2``:	int]
- ``max_epochs``:			Number of epochs to train                                [``74``:	int]
- ``lr``:					Learning rate of the optimizer                           [``0.0005739979509018617``:	float]
- ``training-batch-size``:	Batch size for training batches                          [``10``:	int]
- ``test-batch-size``:		Batch size for test batches                              [``60``:	int]
- ``accelerator``:			Accelerator connecting to the Lightning Trainer          [``'dp'``:	string]
- ``gamma-factor``:			Gamma for the Focal Loss function                        [``2.8074708243593878``:	float]
- ``weight-decay``:			Weight decay for the AdaBelief optimizer                 [``0.08843850663784153``:	float]
- ``epsilon``:				Epsilon for the AdaBelief optimizer                      [``2.1335101419972938e-14``:	float]
- ``alpha-0``:				Alpha for Focal Loss function                            [``0.39228916176765655``:	float]
- ``alpha-1``:				Alpha for Focal Loss function                            [``0.4476337434213018``:	float]
- ``alpha-2``:				Alpha for Focal Loss function                            [``0.8902483094365905``:	float]
- ``alpha-3``:				Alpha for Focal Loss function                            [``0.6695418278351652``:	float]
- ``alpha-4``:				Alpha for Focal Loss function                            [``0.9382751049017035``:	float]
- ``log-interval``:			Number of batches to train for before logging            [``100``:	int]
- ``general-seed``:			Python, Random, Numpy seed                               [``0``:	int]
- ``pytorch-seed``:			Pytorch specific seed                                    [``0``:	int]
