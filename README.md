# fewshot_neuroimaging_classification

This repository contains the code of the experiments performed in the following paper\
[Few-shot Learning for Decoding Brain Signals](https://arxiv.org/pdf/2010.12500.pdf)\
by Myriam Bontonou, Nicolas Farrugia and Vincent Gripon

## Abstract
Few-shot learning consists in addressing data-thrifty (inductive few-shot) or label-thrifty (transductive few-shot) problems. So far, the field has been mostly driven by applications in computer vision. In this work, we are interested in stressing the ability of recently introduced few-shot methods to solve problems dealing with neuroimaging data, a promising application field. To this end, we propose a benchmark dataset and compare multiple learning paradigms, including meta-learning, as well as various backbone networks. Our experiments show that few-shot methods are able to efficiently decode brain signals using few examples, and that graph-based backbones do not outperform simple structure-agnostic solutions, such as multi-layer perceptrons.

## Usage
### 1. Dependencies
- Python = 3.6 
- Pytorch = 1.3

### 2. Dataset construction
The original IBC dataset (releases 1 and 2) [1] is accessible on NeuroVault [2] (collection 6618). In the dataset folder, we propose a notebook to discover the data and a notebook to explaining how we adapt the IBC dataset to create a benchmark dataset for few-shot learning. You have to execute them to download the data from NeuroVault.

### 3. Few-shot learning paradigms
Below, we describe how to reproduce our results using either SimpleShot (inductive few-shot method), MAML_plus_plus (inductive few-shot learning) or PT + MAP (trasnductive few-shot learning).

Our code is built upon the original codes of ["How to train your MAML"](https://openreview.net/pdf?id=HJGven05Y7) (MAML_plus_plus) and ["SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning"](https://arxiv.org/pdf/1911.04623.pdf) (SimpleShot). We adapted the original codes to work on new backbone networks and on new data. 

By using this code, you agree to the [license](https://github.com/mbonto/fewshot_neuroimaging_classification/blob/main/LICENSE) file of this work and to the license files of the original codes of [MAML_plus_plus](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch) and [SimpleShot](https://github.com/mileyan/simple_shot).

#### SimpleShot [3]
Go to the SimpleShot folder. To train a backbone and evaluate the SimpleShot method on few-shot problems, have a look at the Train_and_evaluate notebook.
To evaluate the performance of a nearest class mean classifier applied directly on the inputs, have a look at the Baseline_NCM notebook. To evaluate the performance of a logistic regression trained directly on the inputs, have a look at the Baseline_LR notebook.

#### PT + MAP [4]
Go to the SimpleShot folder. PT+MAP proposes a different way of solving few-shot problems based on the knowledge of other unlabeled samples (transductive setting). In this article, it relies on the backbone trained with SimpleShot. To evaluate PT+MAP method, have a look at the Train_and_evaluate notebook.

#### MAML++ [5]
Go to the MAML_plus_plus folder. To train a backbone and evaluate the MAML++ method, execute the following commands:

0- Dataset.\
To execute the MAML++ code, all images need to be stored in a folder in a particular way.
Thus, to prepare the data, go to the datasets folder and execute the notebook Format_IBC.

1- Define the configuration of the experiment.\
Go to the experiment_config folder and execute the notebook Create config file.

2- Generate a script to launch the experiment.\
Go to script_generation_tools and execute the command: `python generate_scripts.py`.

3- Initialize the folder where results will be stored.\
Execute the notebook create_results_file.

4- Launch the training and the evaluation.\
Go to experiment_scripts and launch the scripts you have defined before. For instance,
`./IBC_5_way_1_shot_0_seed_GNN_model_1_depth_1024_features_few_shot.sh`. (You may need to authorize the execution of the script by executing the command
`chmod 777 IBC_5_way_1_shot_0_seed_GNN_model_1_depth_1024_features_few_shot.sh`.)

5- Results.\
Go to the results folder to see the results.

### 4. Backbone networks
We experiment three architectures of backbone networks: a multi-layer perceptron (MLP), a graph neural network (GNN) and a convolutional neural network (CNN).

- MLP: the input is seen as a simple vector describing the activity of the considered regions of interest. *We vary 2 hyperparameters: the number of hidden layers and the number of features per hidden layer.*

- GNN: the input is the vector described before. It is diffused once on a graph describing the interaction between regions of interest, and then it is handled by a MLP. *We vary the same hyperparameters as a MLP.* Here are some precisions about the diffusion on  the graph:
     - In this work, the graph is a thresholded structural graph (1% highest connection weights), previously introduced in [6]. It has been obtained with tractography, a method measuring the strength of the white fibers between regions in the brain.
     - Denoting A the adjacency matrix of the structural graph (such that A[i,j] represents the strength of the connections between region i and region j and A[i,i] = 0), we define the diffusion of an input vector on the structural graph as: the matrix multiplication of the input vector with D̂<sup>-1/2</sup>ÂD̂<sup>-1/2</sup>, where Â = A + I and D̂ is the degree matrix of Â.

- CNN: 1x1 convolutional neural network, considering each region of interest independently up to the final layer of the architecture. The last layer is fully-connected. Before the last layer, the feature maps are averaged per input region. *We vary the number of hidden layers and the number of feature maps per hidden layer.*

### 5. Results
Here are the results obtained with two random splits (average accuracy and 95% confidence interval over 10,000 5-way tasks from the novel dataset). The first table is obtained with the split stored in dataset/split1. The second table is obtained with the split stored in dataset/split2. *NCM* stands for nearest-class mean classifier and *LR* for logistic regression. Looking at the baselines, we observe that the few-shot tasks generated from the novel classes of split1 are easier than the ones generated from split2. That explains why figures between both tables are really different.

|  Method  | Setting | Backbone | # hidden layers / features<br>(or feature maps for CNN)|    5-shot  |    1-shot   |
|:--------:|:-------:|:--------:|:---------------------------------------------------------------------:|:-----------:|:-----------:|
| Baseline NCM |   -     |    -     |                            -                         |70.56 +- 0.21|57.26 +- 0.20|
| Baseline LR |   -     |    -     |                            -                         |71.67 +- 0.21|53.04 +- 0.20|
|SimpleShot|Inductive|MLP       |                           2/360                      |**86.00 +- 0.16** |**72.54 +- 0.20**|
|          |         |GNN       |                           2/1024                     |85.14 +- 0.16|71.96 +- 0.21|
|          |         |CNN       |                           2/64                       |74.26 +- 0.20|59.98 +- 0.20|
|PT+MAP    |Transductive|MLP    |                           2/360                      |**88.76 +- 0.17**|**84.34 +- 0.25**|
|          |         |GNN       |                           2/1024                     |87.86 +- 0.18|83.19 +- 0.25|
|          |         |CNN       |                           2/64                       |74.82 +- 0.20|63.71 +- 0.26|
|MAML++    |Inductive|MLP       |                           1/360                      |82.31 +- 0.18|67.64 +- 0.22|
|          |         |GNN       |                           1/128                      |80.87 +- 0.18|67.71 +- 0.22|
|          |         |CNN       |                           2/360                      |76.80 +- 0.22|64.68 +- 0.21|


|  Method  | Setting | Backbone | # hidden layers / features<br>(or feature maps for CNN)|    5-shot  |    1-shot   |
|:--------:|:-------:|:--------:|:---------------------------------------------------------------------:|:-----------:|:-----------:|
| Baseline NCM |   -     |    -     |                            -                         |64.23 +- 0.18|45.59 +- 0.18|
| Baseline LR |   -     |    -     |                            -                         |65.79 +- 0.19|41.35 +- 0.17|
|SimpleShot|Inductive|MLP       |                           2/256                      |**74.1 +- 0.18**|**57.97 +- 0.21**|
|          |         |GNN       |                           1/128                      |73.53 +- 0.18|55.03 +- 0.20|
|          |         |CNN       |                           2/256                      |68.75 +- 0.19|49.82 +- 0.20|
|PT+MAP    |Transductive|MLP    |                           2/256                      |**75.31 +- 0.20**|**65.01 +- 0.29**|
|          |         |GNN       |                           1/128                      |74.34 +- 0.20|63.02 +- 0.29|
|          |         |CNN       |                           2/256                      |73.05 +- 0.21|59.13 +- 0.28|
|MAML++    |Inductive|MLP       |                           1/360                      |71.25 +- 0.18|49.74 +- 0.19|
|          |         |GNN       |                           1/128                      |67.65 +- 0.18|49.94 +- 0.19|
|          |         |CNN       |                           2/128                      |69.32 +- 0.21|54.21 +- 0.21|



## References
[1] [Pinho, A.L. et al. (2020) Individual Brain Charting dataset extension, second release of high-resolution fMRI data for cognitive mapping.](https://project.inria.fr/IBC/ibc-in-a-nutshell/)

[2] [Gorgolewski, K.J. et al. (2015) NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the brain.](https://neurovault.org/)

[3] [Wang, Y. et al. (2019) Simpleshot: Revisiting nearest-neighbor classification for few-shot learning.](https://arxiv.org/pdf/1911.04623.pdf)

[4] [Hu, Y. et al. (2020) Leveraging the Feature Distribution in Transfer-based Few-Shot Learning.](https://arxiv.org/pdf/2006.03806.pdf)

[5] [Antoniou, A. et al. (2018) How to train your MAML.](https://openreview.net/pdf?id=HJGven05Y7)

[6] [Preti, M. G. et al.  (2019) Decoupling of brain function from structure reveals regional behavioral specialization in humans.](https://www.nature.com/articles/s41467-019-12765-7)

## Contact
Please contact us if there are any problems.

Myriam Bontonou (myriam.bontonou@imt-atlantique.fr)
