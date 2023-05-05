from keras.preprocessing.image import ImageDataGenerator
from HelperFiles.ImagePreprocessor import ImagePreProcessing
from pathlib import Path

import numpy as np
import cv2
import PIL
import glob
import os


class HelperClass():

    def load_files_from_dir_supervised(self, directory, width=128, height=128):
        """
        Description
        -----------
            Load image files from a specified directory and assign labels based on the subfolder names.

        This function expects the given directory to contain subfolders 'occupied' and 'unoccupied'.
        Images in the 'occupied' folder will be assigned a label of 1, and images in the 'unoccupied'
        folder will be assigned a label of 0. The images are resized to 128x128 pixels.

        Parameters
        ----------
        directory (str): The path to the directory containing the subfolders with images.

        Returns
        -------
        X (array-like): A NumPy array containing the loaded and resized images.
        y (array-like): A NumPy array containing the corresponding labels for the loaded images.

        Raises
        ------
        ValueError: If the specified directory does not exist or does not contain any subfolders.
        """

        directory = Path(directory)
        # Check if the directory exists
        if not directory.exists():
            raise ValueError(
                'The directory `{}` does not exist.'.format(directory))

        subfolders = []
        for folder in directory.iterdir():
            folder_name = str(folder.name).lower()
            if folder.is_dir() and folder_name == 'occupied' or folder_name == 'unoccupied':
                subfolders.append(folder)

        # Raise ValueError if there are no subfolders
        if not subfolders:
            raise ValueError(
                f"Found unexpected sub-directories. Expected 'occupied' and 'unoccupied' in `{directory}`.")

        images, labels = [], []
        print('Loading Image Data from `{}`..'.format(directory))
        for f in subfolders:
            # Assign a label of 1 if the folder is named 'occupied', otherwise
            # 0
            label = 1 if f.name == 'occupied' else 0
            for filename in sorted(f.iterdir()):
                filepath = str(f / filename)
                image = cv2.imread(filepath)
                if image is None:
                    print(f'Failed to read image `{filepath}`. Skiping..')
                    continue
                image = cv2.resize(image, (width, height))
                images.append(image)
                labels.append(label)
            print(f'Images from {folder} loaded.')

        # Convert the lists of images and labels to NumPy arrays
        X = np.array(images)
        y = np.array(labels)
        print('Training Data Loaded.\n')
        return X, y

    def preprocess_files_from_dir(self, directory):
        """
        Description
        -----------
            Preprocess images from a given directory and return preprocessed data and labels.

        This function loads image files and their corresponding labels from the specified directory
        and applies full preprocessing to each image using the ImagePreProcessing class. The resulting
        preprocessed images and labels are returned as NumPy arrays.

        Parameters
        ----------
        directory (str): The path to the directory containing the image files and their corresponding labels.

        Returns
        -------
        preprocessed_X (array-like): The preprocessed image data as a NumPy array.
        y (array-like): The corresponding labels for the image data as a NumPy array.

        Notes
        -----
        This function assumes that the image files and their corresponding labels are organized
        in a specific way, as defined by the 'load_files_from_dir_supervised' method.
        """
        X, y = self.load_files_from_dir_supervised(directory)
        print('Pre-processing Training Data..')
        preprocessor = ImagePreProcessing().full_preprocess
        preprocessed_X = np.asarray(list(map(preprocessor, X)))
        print('Training Data Preprocessed.\n')

        return preprocessed_X, y

    def rename_files(self, directory, name):
        """
        Description
        -----------
            Renames all files in the specified directory to a new name with incremental numbers.
        Parameters
        ----------
            directory (str): Path of the directory containing the files to be renamed.
            Defaults to current working directory if not provided.

            name (str): New name for the files. Raises ValueError if not provided.
          Raises
          ------
            FileNotFoundError: If source directory does not exist.
            ValueError: If the target file name was not provided.
          """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError('Source directory does not exist.')
        if not name:
            raise ValueError('Target file name was not provided.')
        for i, file in enumerate(directory.iterdir(), start=1):
            if not file.is_file():
                continue
            orig_name = file.name
            new_name = f'{name}_{i}{file.suffix}'
            file.rename(directory / new_name)
            print('Renamed file - "{}"\tto\t"{}"'.format(orig_name, new_name))
        print('\tAll Files Renamed')

    def check_dataset_images_integrity(self, directory='') -> list:
        """
        Description
        -----------
            Check the integrity of all images in a directory and return a list of corrupt images.

        Parameters
        ----------
            directory (str): Directory path to check.

        Raises
        ------
            ValueError: If the specified directory path does not exist.

        Returns
        -------
            corrupt_images (array-like): A list of file paths of corrupt images.

        """
        # Check if the directory exists
        if not os.path.exists(directory):
            raise ValueError('Directory path does not exist.')

        # Initialize a list to hold corrupt images
        corrupt_images: list = []

        # Loop over all images in the directory
        for image in glob.glob(os.path.join(directory, '*.png')):
            try:
                # Open the image and check if it can be read
                PIL.Image.open(image)
            except PIL.UnidentifiedImageError:
                # If the image is corrupt, add its path to the list
                corrupt_images.append(image)

        # Return the list of corrupt images
        print('List of possibly corrupt images being returned..')
        return corrupt_images

    def augmentedImageGenerator(
            self,
            generator,
            src_dir,
            target_dir,
            batch_size=8,
            num_images=4):
        """
        Description
        -----------
            Generate and save augmented images from png images in a directory using Keras' ImageDataGenerator.

        Parameters
        ----------
            generator (ImageDataGenerator): Keras ImageDataGenerator object.
            src_dir (str): Directory containing the png images.
            target_dir (str): Directory to save the augmented images.
            batch_size (int): Size of the batch of augmented images to generate.
            num_images (int): Number of augmented images to generate for each input image.

        Raises
        ------
            FileNotFoundError: If source directory or target directory does not exist.
            ValueError: If generator is not an instance of ImageDataGenerator.
        """
        # Check if the source directory and target directory exist
        if not os.path.exists(src_dir):
            raise FileNotFoundError("Source Directory does not exist.")
        # If there is no images (png format) in the source folder
        if len(glob.glob(os.path.join(src_dir, '*.png'))) < 1:
            print('No images in source directory.')
            return
        # If the target directory does not exist, create it
        if not os.path.exists(target_dir):
            print('Target Directory does not exist, creating directory..')
            os.makedirs(target_dir)
            print('Target Directory Created.')

        # Check if generator is an instance of ImageDataGenerator
        if not isinstance(generator, ImageDataGenerator):
            raise ValueError(
                "'generator' is not an instance of ImageDataGenerator.")

        # Get the name of the target directory
        target_folder_name = os.path.basename(target_dir)

        # Loop through all the png images in the source directory
        for image in glob.glob(os.path.join(src_dir, '*.png')):
            # Load the image and convert it to a numpy array
            img: PIL = cv2.imread(image)
            img_array: np.array = np.asarray(img)
            # Add an additional dimension to the numpy array
            img_array: np.array = img_array.reshape((1,) + img_array.shape)

            i: int = 0
            # Print a message indicating that the current image is being
            # transformed
            print(f'Transforming image - {os.path.basename(image)}')
            # Generate images with random transformations and save them in the
            # target directory
            for batch in generator.flow(
                    img_array,
                    batch_size=batch_size,
                    save_to_dir=target_dir,
                    save_prefix=target_folder_name,
                    save_format='png'):
                i += 1
                # If the desired number of images have been generated, break
                # out of the loop
                if i > num_images:
                    break
