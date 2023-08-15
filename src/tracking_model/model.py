import os
import numpy as np
import tifffile as tiff
import cv2
from PIL import Image
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Read tiff file
def read_tif_file(filepath):

    # Read and load volume
    tiff_image = tiff.imread(filepath)

    # Convert to a numpy array
    numpy_array = np.array(tiff_image)

    # Transpose
    transposed_image = numpy_array.transpose(2, 3, 0, 1)

    # Print the shape
    print("Loaded image Shape:", transposed_image.shape)

    return transposed_image

# Convert the prediction numpy array to TIFF format
def save_tif_file(image_obj, file_path):
    # Transpose the numpy array back to its original shape
    transposed_image = image_obj.transpose(2, 3, 0, 1)

    # Save the numpy array as a TIFF file
    tiff.imwrite(file_path, transposed_image)
    
    # Print location
    print("Predicted image saved at: ", file_path)

# Process image
def process_image(filepath):

    # Read file
    processed_image = read_tif_file(filepath)

    return processed_image

# Function for resizing the images
def resize_images(images, target_shape):
    num_images = images.shape[0]
    resized_images = np.zeros((num_images,) + target_shape, dtype=images.dtype)

    for i in range(num_images):
        # Calculate the resize factors for each dimension
        resize_factors = [target_shape[dim] / images.shape[1:4][dim] for dim in range(3)]
        resized_images[i] = zoom(images[i], resize_factors + [1], order=1)

    return resized_images

# Layers of the model
def get_model(width=128, height=128, depth=128, channels=2):
    """Build a 3D convolutional neural network model."""

    inputs = layers.Input((width, height, depth, channels))

    # 3D convolutional neural network layers
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)

    # Increase spatial resolution
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D(size=(2, 2, 2))(x)

    x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D(size=(2, 2, 2))(x)

    # Output layer
    outputs = layers.Conv3D(channels, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# Train model 
def model_train(path_original, path_groundtruth, path_outfolder):
    
    #print("Tensorflow: ", tf.__version__)
    
    # Folder "original" consists of the original unlabelled images
    original_images_paths = [
        os.path.join(os.getcwd(), path_original, x)
        for x in os.listdir(path_original)
    ]

    # Folder "groundtruth" consists of the labelled images
    labelled_images_paths = [
        os.path.join(os.getcwd(), path_groundtruth, x)
        for x in os.listdir(path_groundtruth)
    ]
    print("Number of images: " + str(len(original_images_paths)))
    
    # Read and process the images
    original_images = np.array([process_image(path) for path in original_images_paths])
    labelled_images = np.array([process_image(path) for path in labelled_images_paths])
    
    # Target shape
    target_shape = (128, 128, 128, 2)

    # Resize the images
    resized_sm_original_images = resize_images(original_images, target_shape)
    resized_sm_labelled_images = resize_images(labelled_images, target_shape)
    print("Resized images shape: " + str(resized_sm_original_images.shape))
    
    # Split data in the ratio 70-30 for training and validation.
    total_images = len(resized_sm_original_images)
    split_point = int(0.7 * total_images)  # 70% of the total images

    x_train = np.stack((resized_sm_original_images[:split_point]), axis=0)
    y_train = np.stack((resized_sm_labelled_images[:split_point]), axis=0)
    x_val = np.stack((resized_sm_original_images[split_point:]), axis=0)
    y_val = np.stack((resized_sm_labelled_images[split_point:]), axis=0)
    print(
        "Number of samples in train and validation are %d and %d."
        % (x_train.shape[0], x_val.shape[0])
    )
    
    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    # Create the model
    model = get_model(width=128, height=128, depth=128, channels=2)
    model.summary()
    
    batch_size = 2
    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        # .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        # .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )

    # Train model
    # Compile model
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Define callbacks.
    weights_file_loc = os.path.join(path_outfolder, "weights_file.h5")
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        weights_file_loc, save_best_only=True
    )
    #early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    # Train the model, doing validation at the end of each epoch
    epochs = 10 #100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb], #, early_stopping_cb],
    )
    
    # Save the model
    model_file_loc = os.path.join(path_outfolder, "trained_model.h5")
    model.save(model_file_loc)

    return 

# Use model
def model_predict(image_path, trained_model_file, weights_file, path_outfolder):
    
    # Read and process the images
    original_image = np.array([process_image(image_path)])
    
    # Target shape
    target_shape = (128, 128, 128, 2)

    # Resize the images
    resized_sm_original_image = resize_images(original_image, target_shape)
    print("Resized image shape: " + str(resized_sm_original_image.shape))
    
    # Load model.
    trained_model = tf.keras.models.load_model(trained_model_file)
    
    # Load best weights.
    trained_model.load_weights(weights_file)

    # Predict
    prediction = trained_model.predict(np.expand_dims(resized_sm_original_image[0], axis=0))[0]
    
    print("Prediction shape: ", prediction.shape)
    
    # Save the prediction as a TIFF file
    predictedfile_loc = os.path.join(path_outfolder, "predicted_image.tif")
    save_tif_file(prediction, predictedfile_loc)
    
    return

################## Run ##################
def run_train_model(og_folder: os.PathLike, gt_folder: os.PathLike, outfolder: os.PathLike):

    print("Train model...\n")
    model_train(og_folder, gt_folder, outfolder)
    
    return outfolder

def run_predict_model(test_image: os.PathLike, model_file: os.PathLike, weights_file: os.PathLike, outfolder: os.PathLike):
    
    print("Predict with the model...\n")
    model_predict(test_image, model_file, weights_file, outfolder)
    
    return outfolder

#run_train_model("/home/eadam002/rse/prepare_data/cropped_images/original", "/home/eadam002/rse/prepare_data/cropped_images/groundtruth", "/home/eadam002/rse/packaging_tests/model_versions/test_data")

#run_predict_model("/home/eadam002/rse/prepare_data/cropped_images/original/imagepair_076_077.tif", "/home/eadam002/rse/packaging_tests/model_versions/test_data/trained_model.h5", "/home/eadam002/rse/packaging_tests/model_versions/test_data/weights_file.h5", "/home/eadam002/rse/packaging_tests/model_versions/test_data")


