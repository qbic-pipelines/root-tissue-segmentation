FROM mlfcore/base:1.2.0

# Install the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a

# Activate the environment
RUN echo "source activate root_tissue_segmentation" >> ~/.bashrc
ENV PATH /home/user/miniconda/envs/root_tissue_segmentation/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name root-tissue-segmentation > root-tissue-segmentation_environment.yml

# Currently required, since mlflow writes every file as root!
USER root
