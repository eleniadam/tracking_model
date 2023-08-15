# tracking_model: Track cells across timestamped 3D images

## Installation 

### Install on your own machine

You can run the following commands to install the tool in your own conda environment.

#### Windows Install

1. Download and install **Python 3.9** version of Miniconda for Windows: https://docs.conda.io/en/latest/miniconda.html#windows-installers

2. **Login**, download and install Visual Studio 2022 Professional to build pyklb: https://visualstudio.microsoft.com/vs 

3. Open "Command Prompt" and create a conda environment and activate it:
```
conda create -n trackcells python=3.9
conda activate trackcells
```

4. Install the tracking_model:
```
pip install git+https://github.com/eleniadam/tracking_model.git
track_cell --help
```

#### Example

1. Crop the 3D images using roi_convertor


2. Relabel the timestamp i+1 3D image (according to timestamp i 3D image): 

**Commandline Options**

```track_cell generate-relabelledimage --help```

**Example Command** 

```
track_cell generate-relabelledimage 
--tree_file edges_1_100_st6.csv 
--image_i klbOut_Cam_Long_00084.lux_SegmentationCorrected.klb 
--image_ii klbOut_Cam_Long_00085.lux_SegmentationCorrected.klb 
--output_dir out_data
```

The CSV file must depict each tree edge in the form: 
timestampN_labelA,timestampM_labelB
where timestamp{N,M} and label{A,B} are three digit numbers.

image_i, image_ii can be in klb/h5/tif/npy formats with these extensions and the filename must contain the three digit number of the timestamp.

Output image will be in tif format and saved as relabel_timestampii.tif


3. Superimpose two 3D images into one 4D image:

**Commandline Options**

```track_cell generate-superimposedimage --help```

**Example Command** 
```
track_cell generate-superimposedimage 
--image_i klbOut_Cam_Long_00084.lux_SegmentationCorrected.klb 
--image_ii klbOut_Cam_Long_00085.lux_SegmentationCorrected.klb 
--output_dir out_data
```

image_i, image_ii can be in klb/h5/tif/npy formats with these extensions.

Output 4D image will be in tif format and saved as imagepair_timestampi_timestampii.tif

4. Train the model:

**Commandline Options**

```track_cell train-model --help```

**Example Command** 
```
track_cell train-model 
--original_dir original 
--groundtruth_dir groundtruth 
--output_dir out_data
```

Input 4D images must be in tif format. 
The output folder contains the model and optimal weights information saved as trained_model.h5 and weights_file.h5

5. Predict the image:

**Commandline Options**

```track_cell predict-image --help```

**Example Command** 
```
track_cell predict-image 
--image imagepair_076_077.tif 
--model_file trained_model.h5 
--weights_file weights_file.h5 
--output_dir out_data
```

Input 4D image must be in tif format.
The model and weights files are the output files of the train-model command.


