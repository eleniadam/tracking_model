import os
import re
import numpy as np
import tifffile as tiff
from .GeometricInterpolation import InterpolateTubes, DrawContourForLabel
from .io_utils_el import write_image_4d
from .Label_read import *

# Read two input tiff images and save into numpy arrays (io-utils)
# Superimpose images
# Output a numpy array
# Write output tiff image from the numpy array (io-utils)

##############################################################
# Function to superimpose two 3D images into one
# Input:
# image_timestamp_i - 3D image timestamp i
# image_timestamp_ii - 3D image timestamp i+1
# output_folder - location to save the output superimposed image
# Output:
# Superimposed image tiff file
##############################################################
def create_superimposed_image(image_timestamp_i, image_timestamp_ii, output_folder):

    # Timestamps
    pattern = r'(\d{3})\b'
    time_index1 = re.findall(pattern, image_timestamp_i)[-1]
    time_index2 = re.findall(pattern, image_timestamp_ii)[-1]
    print('Timestamps:', time_index1, time_index2)

    # Read input images
    image_array1 = read_image(image_timestamp_i)
    image_array2 = read_image(image_timestamp_ii)

    # Identify labels
    labels1 = np.unique(image_array1)
    nlabels1 = len(labels1) - 1
    labels2 = np.unique(image_array2)
    nlabels2 = len(labels2) - 1
    #print('Number of nuclei in images:', nlabels1, nlabels2)
    #print('Labels in image1:', labels1, '\nLabels in image2:', labels2)

    # Superimpose
    # Image with 2 channels, channel1 is value of image1, channel2 is value of image2
    new_shape = np.shape(image_array1) + (2,) 
    array_superimposed = np.zeros(new_shape, dtype=image_array1.dtype) 
    array_superimposed[:,:,:,0] = image_array1
    array_superimposed[:,:,:,1] = image_array2

    labels_s = np.unique(array_superimposed)
    nlabels_s = len(labels_s) - 1
    print('Shape of superimposed image:', np.shape(array_superimposed))
    #print('Labels in superimposed image:', labels_s)
    
    # Save new label image
    new_image_labels_cont = np.ascontiguousarray(array_superimposed)
    
    # Write 4D image
    write_image_4d(new_image_labels_cont, os.path.join(output_folder, "imagepair_" + str(time_index1) + "_" + str(time_index2)), 'TIF')

    print("Superimposed image saved: ", time_index1, "-", time_index2)

################## Run ##################
def run_superimpose(input_image1: os.PathLike, input_image2: os.PathLike, outfolder: os.PathLike):
    
    print("Superimpose...")
    
    create_superimposed_image(input_image1, input_image2, outfolder)
    
    return outfolder


