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
The original IBC data are accessible on NeuroVault. In the dataset folder, we propose a notebook to discover the data and a notebook to adapt the data to the few-shot learning setting. You have to execute them to download the data from NeuroVault and to split them in a random way in three parts (base dataset, validation dataset, novel dataset).

### 3. Few-shot learning paradigms
Our code is built upon the original codes of ["How to train your MAML"](https://openreview.net/pdf?id=HJGven05Y7) (MAML_plus_plus) and ["SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning"](https://arxiv.org/pdf/1911.04623.pdf) (SimpleShot). They have been adapted to work on new backbone networks and on new data.

By using this code, you agree to the [license](https://github.com/mbonto/fewshot_neuroimaging_classification/blob/main/LICENSE) file of this work and to the license files of the original codes of MAML_plus_plus and SimpleShot.

#### SimpleShot
Go to the SimpleShot folder. To train a backbone and evaluate the SimpleShot method on few-shot problems, have a look at the Train_and_evaluate notebook.
To evaluate a baseline based on a nearest class mean classifier, have a look at the baseline notebook.

#### PT + MAP
Go to the SimpleShot folder. PT+MAP proposes a different way of solving few-shot problems based on the knowledge of other unlabeled samples (transductive setting). In this article, it relies on the backbone trained with SimpleShot. To evaluate PT+MAP method, have a look at the Train_and_evaluate notebook.

#### MAML++
Go to the MAML_plus_plus folder. To train a backbone and evaluate the MAML++ method, execute the following commands:

0- Dataset.\
To execute the MAML++ code, all images need to be stored in a folder in a particular way.
Thus, to prepare the data, go to the datasets folder and execute the notebook DownloadIBC.

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

### 5. Results
Here are the results obtained with two random splits (average accuracy and 95% confidence interval over 10,000 5-way tasks from the novel dataset).

|  Method  | Setting | Backbone |     5-shot  |    1-shot   |
|:--------:|:-------:|:--------:|:-----------:|:-----------:|
| Baseline |   -     |    -     |65.85 +- 0.18|50.35 +- 0.18|
|SimpleShot|Inductive|MLP       |76.06 +- 0.18|58.68 +- 0.20|
|          |         |GNN       |75.63 +- 0.18|57.58 +- 0.20|
|          |         |CNN       |66.88 +- 0.20|51.37 +- 0.21|
|PT+MAP    |Transductive|MLP    |**78.77 +- 0.20**|**69.35 +- 0.28**|
|          |         |GNN       |**78.71 +- 0.20**|**69.31 +- 0.28**|
|          |         |CNN       |72.05 +- 0.21|60.64 +- 0.29|
|MAML++    |Inductive|MLP       |**77.82 +- 0.19**|**62.86 +- 0.23**|
|          |         |GNN       |**77.63 +- 0.19**|**63.18 +- 0.23**|
|          |         |CNN       |74.43 +- 0.19|59.81 +- 0.21|

|  Method  | Setting | Backbone |     5-shot  |    1-shot   |
|:--------:|:-------:|:--------:|:-----------:|:-----------:|
| Baseline |   -     |    -     | +- | +- |
|SimpleShot|Inductive|MLP       | +- | +- |
|          |         |GNN       | +- | +- |
|          |         |CNN       | +- | +- |
|PT+MAP    |Transductive|MLP    | +- | +- |
|          |         |GNN       | +- | +- |
|          |         |CNN       | +- | +- |
|MAML++    |Inductive|MLP       | +- | +- |
|          |         |GNN       | +- | +- |
|          |         |CNN       | +- | +- |

## References
[Pinho, A.L. et al. (2020) Individual Brain Charting dataset extension, second release of high-resolution fMRI data for cognitive mapping.](https://project.inria.fr/IBC/ibc-in-a-nutshell/)

[Gorgolewski, KJ et al. (2015) NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the brain.](https://neurovault.org/)

[SimpleShot](https://arxiv.org/pdf/1911.04623.pdf)

[PT+MAP](https://arxiv.org/pdf/2006.03806.pdf)

[MAML++](https://openreview.net/pdf?id=HJGven05Y7)

## Contact
Please contact us if there are any problems.

Myriam Bontonou (myriam.bontonou@imt-atlantique.fr)
