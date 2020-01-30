# AURORA - AUtomated classification of metamodel RepOsitories using a neuRAl network.

This repository contains the source code implementation of AURORA and the datasets used to replicate the experimental results of our paper that has been accepted at MODELS'19:

_Automated Classification of Metamodel Repositories: A Machine Learning Approach_

Phuong T. Nguyen, Juri Di Rocco, Davide Di Ruscio, Alfonso Pierantonio<sup>(1)</sup>, Ludovico Iovino<sup>(2)</sup>

<sup>(1)</sup> Università degli Studi dell'Aquila, Italy

<sup>(2)</sup> Gran Sasso Science Institute, Italy

A pre-print version of the paper is available [here](https://github.com/MDEGroup/AURORA/blob/master/MODELS2019.pdf).

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

## Acronym
The name AURORA has a nice connotation. Aurora is northern lights where there are distinctive bands of moving, colorful lights, which somehow resemble separate metamodel categories. Furthermore, in Italian aurora means "the light of a new day."


## Disclaimer

The following [dataset](http://doi.org/10.5281/zenodo.2585431) has been exploited in our evaluation. However, we do not redistribute data from there. We only mine it to produce metadata that can be used as input for AURORA.


## How to cite
If you find our work useful for your research, please cite the paper using the following BibTex entry:

```
@INPROCEEDINGS{8906979,
	author={P. T. {Nguyen} and J. {Di Rocco} and D. {Di Ruscio} and A. {Pierantonio} and L. {Iovino}},
	booktitle={2019 ACM/IEEE 22nd International Conference on Model Driven Engineering Languages and Systems (MODELS)},
	title={Automated Classification of Metamodel Repositories: A Machine Learning Approach},
	year={2019},
	volume={},
	number={},
	pages={272-282},
	keywords={Machine learning;metamodel repositories;metamodel classification},
	doi={10.1109/MODELS.2019.00011},
	ISSN={null},
	month={Sep.},}

```


