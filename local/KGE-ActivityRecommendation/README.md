# KGE-ActivityRecommendation
This is the supplementary reposititory for the paper:
"Knowledge Graph Completion for Activity Recommendation in Business Process Modeling"

# Installation libKGE library
We used the PyTorch-based library [libKGE](https://github.com/uma-pi1/kge) to apply knowledge graph embedding models. Therefore, you need to install libKGE to replicate our study:
1. Clone libkge (i.e., `git clone https://github.com/uma-pi1/kge.git`) to have all the library's contents in the root folder **kge**.
2. Navigate to **kge** folder: `cd kge`

   We recommend to create a python environment using: `python -m venv kge_work` `source kge_work/bin/activate` to avoid any discrepency between libraries.
3. Install all requirements: `pip install -e .`
4. Make yourself familiar with the main functionalities of libKGE using the documentation provided on the [libKGE repository](https://github.com/uma-pi1/kge). 

# Adding datasets, and configuration files
The **kge** folder includes a folder called **local** which will be used for all our experiments (see: https://github.com/uma-pi1/kge/tree/master/local). In order to replicate our results, you need to:
1. Clone this github repository: `git clone https://github.com/keyvan-amiri/KGE-ActivityRecommendation.git`
2. Run `mv KGE-ActivityRecommendation local/` to move the folder **KGE-ActivityRecommendation** (i.e., current repository) into **kge/local**.
3. Navigate to **KGE-ActivityRecommendation** `cd local/KGE-ActivityRecommendation` and run `python file_transfer.py` to add all datasets, and configuration files to the libKGE that you have already cloned.

Current github repository includes the datasets used in our experiments as well as the best configuration files for KGE models. 

All datasets used in our experiment can be found in the folder **bpmai datasets** within current repository, while the following table provides additional information about these datasets:

| Folder name | Corresponding dataset in the paper |
|----------|----------|
| bpmai_lastrev_caise_onlyAfter  | Translation approach **1** without augmentation | 
| bpmai_lastrev_caise_onlyAfter2 | Augmented dataset for translation approach **1**| 
| bpmai_lastrev_caise | Translation approach **2a** without augmentation | 
| bpmai_lastrev_caise2 | Augmented dataset for translation approach **2a** |
| bpmai_lastrev_caise_inProcess | Translation approach **2b** without augmentation | 
| bpmai_lastrev_caise_inProcess2 | Augmented dataset for translation approach **2b** | 

<!-- This is not remaining of the table. -->
All configuration files that are used for training KGE models in our experiment can be found in the folder **configs** within current repository. Similar to the dataset folder, here we have separate subfolders for all datasets used in our experiments.

For training, and evaluation of KGE models you need to copy all dataset folders to the folder **data** within the root folder **kge**. For more information, please refer to ["use your own dataset"](https://github.com/uma-pi1/kge#use-your-own-dataset) on github repository of libKGE. You also need to copy all configuration files to the related folders in the folder **data** within the root folder **kge**. For more information, please refer to ["Using LibKGE"](https://github.com/uma-pi1/kge#using-libkge). To simplify this process, we have provided **file_transfer.py** which takes care of adding all dataasets, and configuration files.

# Preprocessing datasets
Each dataset consists of three files **train.txt**, **valid.txt** and **test.txt** each of them including all relevant triples like this one:

quotationsent_840044706_rev15	after	quotationrecieved_840044706_rev15

In order to train and evaluate KGE models, you need to preprocess these files: first navigate to **kge/data** folder, and then run the preprocessing file while providing the name of the dataset as an argument.

For instance, run `python preprocess_default.py bpmai_lastrev_caise2` for preprocessing the augmented version of translation approach 2a. For more information, see the ["preprocessing file"](https://github.com/uma-pi1/kge/blob/master/data/preprocess/preprocess_default.py) in the libKGE library.

# Training a KGE model
Training a KGE model should be done by specifying training configuratons within a YAML file. All configuration files used in our experiments can be found in the folder **configs** and have been already moved to the related dataset folders in the cloned libKGE repository using the **file_transfer.py**. For more information see ["Using LibKGE"](https://github.com/uma-pi1/kge/tree/master#using-libkge) and ["Defalt configurations"](https://github.com/uma-pi1/kge/blob/master/kge/config-default.yaml) from the documentation provided on libKGE repository.

To train a KGE model, navigate to **kge** folder (i.e. the root folder of cloned libKGE repository) use `kge start` command while providing the path to the relevant YAML configuration file. For instance, to train Distmult model on translation approach 1 simply run:

`kge start data/bpmai_lastrev_caise_onlyAfter/distmult-KvsAll-kl.yaml`

# Performance evaluation of a KGE model
Results of a training job are automatically stored in a separate folder within the **kge/local/experiments** folder. Name of this folder includes the timestamp of the start of the training job, as well as the name of the configuration file. For instance, for above training job we might have a folder name like this: *20231027-145406-distmult-KvsAll-kl* . Apart from log file which can be used to get more insight about the learning process, we are mainly intrested in the best check point file that is saved in this folder. This file always has the same name: **checkpoint_best.pt**. For performance evaluation you only need to navigate to **kge** folder (i.e. the root folder of cloned libKGE repository) and run `kge_evaluation.py`.

The `kge_evaluation.py` file provides the overall results with respect to the evaluation metrics, namely Hits@10 and MRR. It also creates two .text files for all predicted labels in the test set (one for the original predictions, and one for the predictions after post-processing). These fies by default will be saved in the **kge/local/new results** folder. For instance, if we have the following in one of these files:

*searchforpatientsfile_1168188392_rev14 hasLabel searchforpatientsfile

Tails: retrievesubjectdetailsbyid	searchforpatientsfile	notifyemployee ....*

it means that the correct label *searchforpatientsfile* is predicted in the second position, and therefore, if we had only this one example in our test set we would get MRR of 50\% .

