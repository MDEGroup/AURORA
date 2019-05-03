# AURORA - AUtomated classification of metamodel RepOsitories using a neuRAl network.

This repository contains the source code implementation of AURORA and the datasets used to replicate the experimental results of a paper submitted to MODELS'19:

_Automated Classification of Metamodel Repositories: A Machine Learning Approach_


## Introduction

Manual classification methods of metamodel repositories require highly trained personnel and the results are usually influenced by subjectivity of human perception. Therefore, automated metamodel classification is very desirable and stringent. In this work, we apply Machine Learning techniques to automatically classify metamodels. In particular, we implement a tool on top of a feed-forward neural network. An experimental evaluation over a dataset of 555 metamodels demonstrates that the technique permits to learn from manually classified data and effectively categorize incoming unlabeled data with a considerably high prediction rate: the best performance comprehends 95.40% as success rate, 0.945 as precision, 0.938 as recall, and 0.942 as F-1 score. 

## Repository Structure

This repository is organized as follows:

* The [TOOLS](./TOOLS) directory contains the implementation of the different tools we developed:
	* [TERM-EXTRACTOR](./TOOLS/TERM_EXTRACTOR): The Java implementation term extractor from metamodels;
	* [TDM_ENCODER](./TOOLS/TDM-ENCODER): A set of Python scripts allowing to compute TDMs;
	* [NEURAL-NETWORKS](./TOOLS/NEURAL-NETWORKS): This tools classifies metamodels according the TDM values and training set.
* The [DATASET](./DATASET) directory contains the datasets described in the paper that we use to evaluate AURORA:
	* [NORMALIZED_MM_REPRESENTATION](./DATASET/NORMALIZED_MM_REPRESENTATION): plain documents that represent metamodels;
	* [TDMS](./DATASET/TDMS): TDMs are extracted from _NORMALIZE\_MM\_REPRESENTATION_.

## Disclaimer

This [dataset](http://doi.org/10.5281/zenodo.2585431) has been exploited in our evaluation. However, we do not redistribute and data from there. We only mine it to produce metadata that can be used as input for AURORA.
