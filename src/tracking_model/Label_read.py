from .io_utils_el import read_image
import os
import numpy as np
# Reads segmentation results for both membrane and nuclei

# Parameters:
# # membrane_file  = file with membrane segmentation results
# # nuclear_file  = file with nuclear segmentation results
# # nuclear_file_cirrected  = file with hand corrected nuclear segmentation results
# Returns:
# # mem_mask = membrane labels indexed
# # nuc_mask = nuclear labels indexed
# Will return None if one or both the segmentation types is unavailable

def read_segments(data_dir, file_prefix, file_ext, segmentation_type, num_threads:int = 1):
    # Initialize masks to None
    label_mask = None
    # nuclei
    try:
        if segmentation_type == "membrane":
            label_file = construct_membrane_file(data_dir, file_prefix, file_ext)
            if os.path.exists(label_file):
                label_mask = read_image(label_file, num_threads)
        else:
            label_file = construct_nucl_file(data_dir, file_prefix, file_ext)
            if os.path.exists(label_file):
                label_mask = read_image(label_file, num_threads)
    except Exception as e:
            print('Problem with reading segments', e)
    return label_mask


def construct_membrane_file(data_dir, file_prefix, file_ext):
    file_prefix = file_prefix.split(os.extsep)[0]
    mem_file = os.path.join(data_dir, file_prefix + ".crop_cp_masks" + file_ext)
    if os.path.exists(mem_file):
        return mem_file
    else: # Check the other cam
        if 'Long' in file_prefix:
            replace_cam_prefix = file_prefix.replace('Long','Short')
        else:
            replace_cam_prefix = file_prefix.replace('Short','Long')
        mem_file = os.path.join(data_dir, replace_cam_prefix + ".crop_cp_masks" + file_ext)
        if os.path.exists(mem_file):
            return mem_file


def construct_nucl_file(data_dir, file_prefix, file_ext):
    nucl_segm_file = os.path.join(data_dir,file_prefix + ".label" + file_ext)
    nucl_segm_file_corrected = os.path.join(data_dir,file_prefix + "_SegmentationCorrected" + file_ext)
    if os.path.exists(nucl_segm_file_corrected): # Look for corrected mask first
        return nucl_segm_file_corrected
    elif os.path.exists(nucl_segm_file):
        return nucl_segm_file
    else: # Check the other Cam
        if 'Long' in file_prefix:
            replace_cam_prefix = file_prefix.replace('Long','Short')
        else:
            replace_cam_prefix = file_prefix.replace('Short','Long')
        nucl_segm_file = os.path.join(data_dir,replace_cam_prefix + ".label" + file_ext)
        nucl_segm_file_corrected = os.path.join(data_dir,replace_cam_prefix + "_SegmentationCorrected" + file_ext)
        if os.path.exists(nucl_segm_file_corrected): # Look for corrected mask first
            return nucl_segm_file_corrected
        elif os.path.exists(nucl_segm_file):
            return nucl_segm_file


def get_filename_components(image_file_str):
    cur_name = os.path.basename(image_file_str)
    file_prefix = os.path.splitext(cur_name)[0]
    file_ext = os.path.splitext(cur_name)[1]
    file_base = os.path.basename(cur_name).split(os.extsep)
    time_index = int(file_base[0].split('_')[-1])
    return file_base, file_prefix, file_ext, time_index

def filter_timestamp_images(images, min_time, max_time):
    # Set max_time to the len-1 to that we can loop till last
    if max_time == -1:
        max_time = len(images) - 1
    filtered_images = []
    for i, im in enumerate(images):
        image_file_str = str(im)
        file_base, file_prefix, file_ext, time_index = get_filename_components(image_file_str)

        # Check within range to be returned
        if time_index >= min_time and time_index < max_time+1:
            filtered_images.append(image_file_str)
    return np.array(filtered_images)

