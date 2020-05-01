# Data Clustering in Astronomy

This repo contains the code used for analysis in my senior thesis: "Automated Trainable Data Clustering with Applications in Astronomy". The abstract below provides context for the analysis contained in this repo.

My advisor, Peter Melchior, and I have been using this to share and update the latest version of the code used for analysis of different datasets.

## Abstract

One of the most commonly used techniques in data science is clustering: dividing a data set into a certain number of groups so that the points in each group have similar properties. There are many different methods that can be used to cluster data more efficiently and accurately; one such method is Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN). In this work, I researched the applications of this method in astronomy and remote sensing. I found that this method can be used to efficiently cluster hyperspectral images, and produced results for datasets from the Nili Fossae region on Mars that are consistent with existing literature. I also investigated different metrics we can use to cluster data in extragalactic surveys and measured the clustering efficiency of HDBSCAN using training datasets. I used this to tune the parameters and improve upon the clustering result. This thesis demonstrates HDBSCANs ability to produce reliable clustering results, and explores ways in which this algorithm can improved to make the clustering process more autonomous. 

## Requirements
- numpy
- matplotlib
- scipy 
- sklearn
- astropy
- scarlet, https://github.com/pmelchior/scarlet.git
- hdbscan, https://github.com/scikit-learn-contrib/hdbscan.git
- btk, https://github.com/LSSTDESC/BlendingToolKit.git

## Overview of Files

### Data
Most of the data used for my analyses can be found in this folder. The one exception is for the CRISM data, whose file is too large to upload to GitHub, but it can be found at this link: https://pds-geosciences.wustl.edu/.

### Test Files
A lot of code didn't make it into my final thesis, so I have stored the files separately in this folder. 

The test, prims, and clustering files contain an approach that used Prim's method of creating a minimum spanning tree to create a clustering result.

AlphaMetric is a notebook that looks into pruning the minimum spanning tree to create a clustering result. In this, I look at the intra-cluster vs inter-cluster distances produced as a success metric for clustering, producing different results by altering the weighting of spatial information in the dataset. 

The hdbscan-hyperspectral notebook is an inital test at applying HDBSCAN to hyperspectral images. 

btkTest is the initial file used to cluster multi-band images. This notebook is very similar to the one I used for my final analysis, but contains a description of the steps within the notebook.

### 3Band
This notebook was a test on a simple dataset, to check whether HDBSCAN could produce successful clustering results for multi-band images.

### BTK
This contains the code used to generate clustering results in multi-band images produced by btk. The success of the clustering results are measured using the Intersection over Union metric, and this is used to determine parameters to produce optimal clustering results.

### CRISM
This notebook is used for analysis of hyperspectral data from the Nili Fossae region on Mars.  

### CapitolHill
This is used for analysis of hyperspectral data from Capitol Hill. 



