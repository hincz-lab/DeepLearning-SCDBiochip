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

Main Code: the main file script that the user will have to run. 

  * [Workflow_SCDnet.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/Workflow_SCDnet.m) ~ This script starts 
with Phase 1 for detecting and extracting adhered sickle RBCs and then transitions to Phase 2 for counting detected cells based on 
a biophysical classification described in more detail with the reference from above. This script will call the following functions: (A-B) [preprocess_channel.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/preprocess_channel.m), (C) [NN_segmentation.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/NN_segmentation.m), (D-E) [Cell_extraction.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/Cell_extraction.m), (F) [NN_classification.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/NN_classification.m). 

Phase 1: detecting and extracting adhered sickle RBCs. 

 * [Phase1_net.mat](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pretrained%20Networks/Phase1_net.mat) ~ Load weights for the neural network into your MATLAB workspace. This pretrained network is required for segmentating and detecting adhered sickle RBCs in whole channel images. 
  * [preprocess_channel.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/preprocess_channel.m) ~ This script crops 1000 tiles from the large whole channel image with size 15000x5250x3. Then the script resizes each cropped tile to 224x224x3 such that the input data fits in the input layer of the neural network. Finally, the script verifies that each image contains three channels. In some situations depending on the experimental conditions with the microscope, we do deal with grayscale images in both one and three channel depth. Therefore, if an whole microchannel image has one channel, we then copy and concatenate the first channel such that we output three channel depth gray scale image.  
  * [NN_segmentation.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/NN_segmentation.m) ~ Input for this function is the 1000 tiles which are then segmented by the pretrained neural network model `Phase1_net.mat`. This script also binarizes each pixel, where the foreground pixels correspond to the classified adhered sickle RBCs. Furthermore, we also require foreground pixel connectivity to be above 60 pixels, thresholding any small segmented groups of pixels. These small groups of pixels do not correspond to cells but arise from segmenting uncertainty. Overall, the Phase 1 segmentation of small pixel connectivity is not a common theme, yet we conduct this additinoal thresholding step as precaution before the next step on cell extraction.  
  * [Cell_extraction.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/Cell_extraction.m) ~ This functions takes in tiles and binary masks then computes and collects the centroids of each segmented cell for the whole channel.
  * [crop_cell.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/crop_cell.m) ~ Input for this function is the image tiles and the corresponding centroids for each detected adhered sickle RBC. Then the function crops small images from the input tiles with a bounding box centered at the computed centroids. Lastly, the function outputs all of the cropped images, containing single cells. 


Phase 2: biophysical classifcation of sickle RBCs. 
   
   * [Phase2_net.mat](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pretrained%20Networks/Phase1_net.mat) ~ Load weights for the neural network that classifies sickle cells based on their morphogical connection to biophysical and adhesive characterisitcs into your MATLAB workspace. Loading this pretrained network is required before classfying and counting sickle RBCs. 
   * [NN_classification](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/NN_classification.m) ~ Here this function takes each extracted single cell image from process (D-E) and then inputs the data into our pretrained neural network with the ResNet-50 architecture `Phase2_net.mat`. The network will then predict biophysical classes and output `total cell`, `deformable cell`, and `non-deformable cell` counts, while also disregarding any extracted objects mistakenly segmented as adhered sickle RBCs to the functionalized channel with endothelial proteins.
   
## Step by Step Pipeline

To help with debugging and understanding the model and pipeline, there is one main script 
([Workflow_SCDnet.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/Workflow_SCDnet.m)) that calls all of the 
neccsesary functions to complete the pipeline. There is no need for user input in terms of deciding on specific parameter or model specifications. Below this text, we will present a walkthrough for the main script and provide a plethora 
of visualizations, while also running the pipeline step by step to inspect the output at each point. Here is the step by step 
walkthrough for the pipeline: 

### Main Script 
After making sure that all of the functions and neural network weights described in the text above is located in the working directory, then the only necessary script that the user most run is [Workflow_SCDnet.m](https://github.com/hincz-lab/DeepLearning-SCDBiochip/tree/master/Pipeline/Workflow_SCDnet.m). In other words, run the following command: 
```
SCD_pipeline()
function SCD_pipeline()
  ... 
end
```
### A-B: Preprocessing (choose image for analysis) 

Here you will choose the given whole channel image found in the working directory. First you will say yes to the following dialog box 
and find the given image in your directory with the following filename `Trial Image.jpg`. While successfully loading the image, you 
should see the dialog boxes followed by the working directory in sequential order: 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82154480-47e6f700-983c-11ea-9907-e3b438652fcd.png" alt="step2 part 1 img" height="80%" width="80%">
</p>

Then, we will load weights for segmenting the adhered sickle RBCs with the following filename `Phase1_net.mat` along with similar dialog boxes. These dialog boxes allow users with no programming expertise to still easily implement the pipeline. 


<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82154504-71a01e00-983c-11ea-8f4a-b3548a484644.png" alt="step 2 part 2 img" height="80%" width="80%">
</p>


Lastly for this step, we will run the following command 

```
function preprocess_channel()
  ... 
end
```
which will evenly crop the original whole channel image into smaller tiles and resize each respective tile such that they fit into the 
input layer for `Phase 1 net`. 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82154593-ea06df00-983c-11ea-8513-fbca91e8da74.png" alt="step 2 part 3 img" height="80%" width="80%">
</p>

### C: Neural Network for Segmentation (track adhered sickle RBCs)

After we collect all 1000 tiles, we will run the following command 

```
function NN_segmentation()
  ... 
end
```

which will then segment each tile by classifying individual pixels based on the trained categories (see manuscript for more detail). This function detects and distinguishes sickle RBCs adhered to the functionalized channel with endothelial proteins versus other objects. 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82154604-0440bd00-983d-11ea-9fbc-101f39305df6.png" alt="step 3 img" 0height="80%" width="80%">
</p>

### Animation that Illustrates the First Three Steps of the Pipeline
The top plot is the whole channel, consisting of stiched together mosaic images. The green box scanning across the channel corresponds 
to the two bottom plots, where the left and the right plot is the segmented and the binarized versions of the cropped image tile. Within 
the bottom two plots, the red pixels (right plot) and white pixels (left plot) corresponds to the segmented adhered cell mask and the 
binarized adhered cell mask. 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82155779-479f2980-9845-11ea-805e-e1160ebbc458.gif" 
height="100%" width="100%">
</p>


### D-E: Extracting Individual Adhered Cell Images (extract adhered sickle RBCs)

Here we will compute the centroid for each segmented sickle cell adhered to the functionalized wall with endothelial proteins with the 
following function:
```
function Cell_extraction()
  ... 
end
```
Then we will crop smaller images which contain single cells centered at their respective centroids with the following command:
```
function crop_cell()
  ... 
end
```
You should first see the following dialog box
<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82705810-5b340100-9c46-11ea-922d-c019b6a9a8cc.png" 
height="40%" width="40%">
</p>

followed by the below dialog box that requires user input in terms of writing a specific path from their machine so that the pipelinecan save extracted adhered sRBCs (example below is for Windows OS).

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82705849-6dae3a80-9c46-11ea-8814-c16b9b625f63.png" 
height="40%" width="40%">
</p>

After choosing your output path to saving and images, you should also see a popup with counts for the total amount of extracted cells 
from the large whole channel image (see image below). 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82705868-756ddf00-9c46-11ea-8fbb-49d7978ac48c.png" 
height="40%" width="40%">
</p>

Furthermore, this step within the pipeline will write and save extracted images of individual cells under the folder name 
`*\extracted_sRBC\`, then the script will add three more folders with the filenames `*\extracted_sRBC\deformable`, `*\extracted_sRBC\nondeformable`, and `*\extracted_sRBC\other` after classfying and counting each extracted cell.

### F: Neural Network for Biophysical Classification  (classify sickle RBCs)

Here we will load our neural network weights with the following filename `Phase2_net.mat` for classification. 

<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82156569-c4cc9d80-9849-11ea-99ed-4a679126f6bd.png" 
height="80%" width="80%">
</p>

Then we will classify each extracted individual cell image into the following three cateogories: `deformable`, `nondeformable`, and 
`other`, while also counting the total amount of sickle cells and the corresponding nondeformable and deformable cells adhered to 
the functionalized channel. This process is computed with the following command: 
```
function NN_classification()
  ... 
end
```
After running the command, a popup will show the counts for each category (see image below). 
<p align="center">
<img src="https://user-images.githubusercontent.com/61917512/82156580-d7df6d80-9849-11ea-8dcd-ed3bd4fe837a.png" 
height="80%" width="80%">
</p>

### Animation that Illustrates the Final Two Steps in the Pipeline
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


## An Important Note About Using This Code 

If you want to compile this source code, please note that you will need access to the following proprietary computing utilities:

   - MATLAB (recommended R2019a edition)
   - MATLAB Toolboxes:  
     - Image Processing Toolbox
     - Deep Learning Toolbox 
     - Statistics and Machine Learning Toolbox
##

Copyright 2020 Case Western Reserve University.  Patent pending.  All rights reserved.
