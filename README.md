[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hincz-lab/DeepLearning-SCDBiochip/blob/master/Main2_google-collab.ipynb)

# Welcome to Integrating Deep Learning with Microfluidics for Biophysical Classification of Sickle Red Blood Cells 

## Introduction 
A key component of [sickle cell disease](https://en.wikipedia.org/wiki/Sickle_cell_disease) (SCD) morbidity is periodic recurrence of painful [vaso-occlusion](https://en.wikipedia.org/wiki/Vaso-occlusive_crisis) and blood flow alteration, where sickle cell adhesion to the endothelial layer and the stiffness of red blood cell (RBC) membranes are the primary mechanisms. Using adhesion assays and microfluidics could improve insight into the mechanisms of RBC adhesion and explore intriguing sub-populations within sickle cells based on such biophysical properties as deformability, morphology, and adhesive strength. Microfluidics can be designed to probe cellular adhesion by flowing whole blood through micro-channels functionalized with endothelial proteins. For example, SCD-Biochip [[Alapan et al., Translational Research, 173, 74-91, (2016)]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4959913/) is an in-vitro adhesion assay where the bottom of the micro-channel is covered with a layer of an endothelial protein or a mixture of proteins. Then we flow whole blood samples across the micro-channel. The analysis of the data from these experimental approaches has been challenging, with a significant bottleneck being manual counting and categorization of cells within the complex images produced from whole blood samples. 

We present here a high-throughput workflow for tackling these issues, utilizing computer vision like image processing techniques, encoder-decoder models, and convolutional neural networks. This workflow analyzes phase-contrast or bright-field mosaic images stitched together by Olympus CellSense live-cell imaging and analysis software with an Olympus 10x/0.25 long working distance objective lens. The algorithm can output cell counts of total sickle RBCs, deformable RBCs, and non-deformable RBCs adhered to the bottom channel of the SCD Biochip device, where the bottom of the channel is covered with endothelial proteins, and the flow within the channel is dynamic. The algorithm segments each adhered sickle RBC and records detailed information about each cell's position within whole blood and under shear flow. Then, the workflow extracts images of individual adhered sickle RBCs by computing bounding boxes around each located cell, centered around the cell's centroids, and finally inputs each extracted image through a convolutional neural network for biophysical classification. The workflow optionally also outputs generated segmentation masks of the whole microchannel image for later analysis. 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/76152290-d2b4b400-608b-11ea-9e57-af36ea29d922.png" alt="pipeline img" height="80%" width="80%">
</p>

The workflow described above can be found in more detail by searching the following reference: 
Praljak, N.,Iram, Iram, S., Singh, G., Hill, A., Goreke, U., Gurkan, U., & Hinczewski,M., “Integrating deep learning with microfluidics 
for biophysical classification of sickle red blood cells.” (In preparation)

## Getting Started 

To help with debugging and understanding the model and pipeline, there is one main script 
([Main.ipynb](https://github.com/hincz-lab/DeepLearning-SCDBiochip/blob/master/Main.ipynb)) that calls all of the 
neccsesary functions to complete the pipeline. In addition, there is a corresponding Google Colab version
([Main2_google-collab.ipynb](https://github.com/hincz-lab/DeepLearning-SCDBiochip/blob/master/Main2_google-collab.ipynb)).  There is no need for user input in terms 
of deciding on specific parameter or model specifications. Below this text, we will present a walkthrough for the main script with visualizations. Here is the step 
by step walkthrough for the pipeline: 

### Main Script 

### Animation that Illustrates Phase I in the Pipeline
The top plot is the whole channel, consisting of stiched together mosaic images. The green box scanning across the channel corresponds 
to the two bottom plots, where the left and the right plot is the segmented and the binarized versions of the cropped image tile. Within 
the bottom two plots, the red pixels (right plot) and white pixels (left plot) corresponds to the segmented adhered cell mask and the 
binarized adhered cell mask. 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82155779-479f2980-9845-11ea-805e-e1160ebbc458.gif" 
height="100%" width="100%">
</p>


### Animation that Illustrates Phase II in the Pipeline
The top plot is the extracted object during `Phase 1` while the the bottom three plots corresponding to the output classes during `Phase 2`.

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82156725-fdb94200-984a-11ea-9301-dc9e62e135ed.gif" 
height="100%" width="100%">
</p>


## Authors 

The presented algorithm is written by Niksa Praljak, Shamreen Iram, and Gundeep Singh. It is originally a product of the [Hinczewski 
Biotheory Group](http://biotheory.phys.cwru.edu/) in the [Department of Physics](https://physics.case.edu/) and [CASE Biomanufacturing and Microfabrication Laboratory](http://www.case-bml.net/) in the [Department of Mechanical 
and Aerospace Engineering](https://engineering.case.edu/emae/) at [Case Western Reserve University](https://case.edu/).

## Acknowledgments

This work was supported by the Clinical and Translational Science Collaborative of Cleveland, UL1TR002548 from the National Center for Advancing Translational Sciences component of the National Institutes of Health (NIH) and NIH Roadmap for Medical Research, Case-Coulter Translational Research Partnership Program, National Heart, Lung, and Blood Institute R01HL133574 and OT2HL152643, and National Science Foundation CAREER Awards 1552782 and 1651560. We also acknowledge with gratitude the contributions of patients and clinicians at Seidman Cancer Center (University Hospitals, Cleveland).
   
## How to Cite
If you use any portion of this code or software in your research, please cite: 
Praljak N.,Iram S., Singh G., Hill A., Goreke U., Gurkan U., & Hinczewski M., “Integrating deep learning with microfluidics for  
“Integrating deep learning with microfluidics for biophysical classification of sickle red blood cells.” (In preparation)

## Contact 
Please contact Niksa Praljak, niksapraljak1 (at) gmail.com with questions or feedback. 


## Installation
    
To use DeepLearning-SCDBiochip functionality, please install within conda:

```
git clone https://github.com/hincz-lab/DeepLearning-SCDBiochip

pip install -r requirements.txt
```
##

Copyright 2020 Case Western Reserve University.  Patent pending.  All rights reserved.
