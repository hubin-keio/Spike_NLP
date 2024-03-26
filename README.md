# Spike_NLP
Use NLP to study the spike protein in SARS-CoV-2 virus.

## Table of Contents
* [Purpose](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#purpose)

* [Installation](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#installation)

* [Usage](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#usage)

* [Citation](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#citation)

## Purpose


## Installation 
To match the packages found in our conda environment, spike_env, you can run `conda env create -f environment.yml`. After activating the conda environment, the pnlp module can be installed using `pip install -e .`. 

Other requirements:
- NVIDIA GPU

## Usage
We offer more in depth documentation located in the [notebooks](https://github.com/hubin-keio/Spike_NLP/tree/master/notebooks) folder, which we recommend reading for further understanding before usage of the models. 
- `development_notes.ipynb`: Development process of the Spike NLP BERT model.
- `create_db.ipynb`: How to create a db for the Spike NLP BERT model.
- `data_processing_notes.ipynb`: How we processed the data sets for our models.
- `model_runner_notes.ipynb`: How to run the Spike NLP BERT model runner.
- `embedding_notes.ipynb`: How to generate embedding pickles for transfer learning.
- `transfer_learning_model_runner_notes.ipynb`: How to run transfer learning model runners with and without pickles.
- `clustering_notes.ipynb`: How to run/create clustering of our data.
- `lineage_analysis_notes.ipynb`: Connecting more metadata with our clusters, such as Pango lineages.

### Quickstart: Running the NLP BERT Model
1) Creating the Databases
  * Prior to running the NLP BERT Model, the `make_db.py` script must be ran to create the RBD Spike Protein database. This can be ran using the command `python make_db.py` from within the `/results/scripts` folder.
2) Executing
  * There are two different versions of the NLP BERT Model that you can run. One is with DistributedDataParallelization (DDP), which utilizes multiple devices (NVIDIA GPUs in this case) to perform faster training and testing; the other is without DDP. Both of these scripts can be ran from the `src/pnlp/runner` folder.
    * To run model runner with DDP: `torchrun --standalone --nproc_per_node=4 gpu_ddp_runner.py`
      * 4 GPUs on a singular node used in this case, set by `--nproc_per_node=4`
    * To run model runner without DDP: `python runner.py`

## Citation
