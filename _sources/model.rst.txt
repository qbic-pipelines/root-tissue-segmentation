Model
======

U^2-Net for semantic segmentation of fluorescence microscopy images.

Overview
~~~~~~~~~~

The trained model classifies ROIs of plant roots into different zones.

Training and test data
~~~~~~~~~~~~~~~~~~~~~~~~

The training data origins from the PHDFM dataset.
501 images are part of the training dataset and a further 100 can be used for testing.

Model architecture
~~~~~~~~~~~~~~~~~~~~~~

The model is based on `Pytorch <https://pytorch.org/>`_ and `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_.
And was adopted from `U^2-Net <https://github.com/xuebinqin/U-2-Net>`_
For Details see: `<https://arxiv.org/pdf/2005.09007.pdf>`_

Evaluation
~~~~~~~~~~~~~

The model was evaluated on 20% (10000 images) of unseen test data. The loss origins from the test data.
The full training history is viewable by running the mlflow user interface inside the root directory of this project:
``mlflow ui``.

Hyperparameter selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameters were chosen on widely known strong defaults.

1. ``AdaBelief optimizer`` was chosen for strong, general performance.
2. ``OneCycle scheduling policy``
3. ``lr`` automatically determined
3. ``lr`` automatically determined
