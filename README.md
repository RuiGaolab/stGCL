# stGCL: A versatile spatial transcriptomics cross-modality fusion method based on multi-modal graph contrastive learning for spatial transcriptomics
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8185216.svg)](https://zenodo.org/record/8185216)


![](https://github.com/RuiGaolab/stGCL/blob/main/stGCL_Overview.png)


## Overview
stGCL is a graph contrastive learning-based cross-modality fusion model that incorporates transcriptional profiles, histological profiles and spatial information to learn the spot joint embedding, enabling spatial domain detection, multi-slices integration and downstream comparative analysis. stGCL utilizes a novel H-ViT method to extract the latent representation of each spot image. Then stGCL employs multi-modal graph attention auto-encoder (GATE) and contrastive learning to extract discriminative information from each modality and fuse them efficiently to generate meaningful joint embeddings. Specifically, multi-modal GATE learns spot joint structured embedding by iteratively aggregating gene expression features and histological features from adjacent spots. In contrastive learning, stGCL maximizes the mutual information between each spot joint embedding and the global summary of the graph, which endows the learned joint representation with not only local (spot expression patterns) but also global features (tissue microenvironment patterns). Furthermore, with a pioneering spatial coordinate correcting and registering strategy, stGCL can precisely identify cross-sectional domains while reducing batch effects.

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
[https://stgcl-turorials.readthedocs.io/en/latest/index.html](https://stgcl-turorials.readthedocs.io/en/latest/index.html)

## Contact
Feel free to submit an issue or contact us at gaorui@sdu.edu.cn for problems about the packages.

## Citation
Na Yu, Daoliang Zhang, Wei Zhang, Zhiping Liu, Xu Qiao, Chuanyuan Wang, Miaoqing Zhao, Baoting Chao, Wei Li, Yang De Marinis, and Rui Gao. stGCL: A versatile spatial transcriptomics cross-modality fusion method based on multi-modal graph contrastive learning for spatial transcriptomics. 2023. (submitted).
