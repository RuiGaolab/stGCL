# stGCL: Spatial domain identification and multi-slice integration analysis for spatial transcriptomics with multi-modal graph contrastive learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8137326.svg)](https://doi.org/10.5281/zenodo.8137326)


![](https://github.com/RuiGaolab/stGCL/blob/main/stGCL_Overview.png)


## Overview
stGCL is a multi-modal graph contrastive learning framework that integrates transcriptional profiles, histological profiles and spatial information learning the spot joint embedding, enabling spatial domain detection, spatial data integration and downstream analysis. First, stGCL utilizes the proposed H-ViT model to extract the latent representation of each spot image. Then stGCL employs multi-modal graph attention auto-encoder (GATE) and contrastive learning to extract discriminative information from each modality and fuse them efficiently to generate meaningful joint embeddings. Specifically, multi-modal GATE learns spot joint structured embedding by iteratively aggregating gene expression features and histological features from adjacent spots. In contrastive learning, stGCL maximizes the mutual information between each spot joint embedding and the global summary of the graph, which endows the learned joint representation with not only local (spot expression patterns) but also global features (tissue microenvironment patterns). Finally, stGCL offers simple and effective vertical and horizontal integration methods for analyzing multiple tissue slices, which encourages smoothing of adjacent spot features within and across slices and mitigates the batch effect.

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.7
* torch~=1.11.0
* pandas~=1.3.5
* numpy~=1.21.5
* scikit-learn~=1.0.2
* scipy~=1.7.3
* seaborn~=0.12.0
* matplotlib~=3.5.2
* scanpy~=1.9.1
* munkres~=1.1.4
* tqdm~=4.64.1
* torchvision~=0.12.0
* anndata~=0.8.0
* opencv-python~=4.6.0.66
* glob2~=0.7
* Pillow~=9.2.0
* setuptools~=63.4.1
* joypy~=0.2.6
* scvelo~=0.2.5
* squidpy~=1.2.2
* stlearn~=0.4.8
* R==4.2.1

## Tutorial
For the step-by-step tutorial, please refer to:
[https://deepst-tutorials.readthedocs.io/en/latest/](https://deepst-tutorials.readthedocs.io/en/latest/)

## Contact
Feel free to submit an issue or contact us at gaorui@sdu.edu.cn for problems about the packages.

## Citation
Na Yu, Daoliang Zhang, Zhi-Ping Liu, Xu Qiao, Wei Zhang, Miaoqing Zhao, Baoting Chao, Chuanyuan Wang, Wei LI, Yang De Marinis, and Rui Gao. stGCL: Spatial domain identification and multi-slice integration analysis for spatial transcriptomics with multi-modal graph contrastive learning. 2023. (submitted).
