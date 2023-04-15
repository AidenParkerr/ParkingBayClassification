                                                                                                                                                                                                                    
                                 ,,          ,,                                                                               ,,                                                        ,,                          
`7MMM.     ,MMF'               `7MM          db                          `7MMF'                                               db                           `7MM"""Mq.                   db                    mm    
  MMMb    dPMM                   MM                                        MM                                                                                MM   `MM.                                        MM    
  M YM   ,M MM   ,6"Yb.  ,p6"bo  MMpMMMb.  `7MM  `7MMpMMMb.  .gP"Ya        MM         .gP"Ya   ,6"Yb.  `7Mb,od8 `7MMpMMMb.  `7MM  `7MMpMMMb.  .P"Ybmmm       MM   ,M9 `7Mb,od8 ,pW"Wq.`7MM  .gP"Ya   ,p6"bo mmMMmm  
  M  Mb  M' MM  8)   MM 6M'  OO  MM    MM    MM    MM    MM ,M'   Yb       MM        ,M'   Yb 8)   MM    MM' "'   MM    MM    MM    MM    MM :MI  I8         MMmmdM9    MM' "'6W'   `Wb MM ,M'   Yb 6M'  OO   MM    
  M  YM.P'  MM   ,pm9MM 8M       MM    MM    MM    MM    MM 8M""""""       MM      , 8M""""""  ,pm9MM    MM       MM    MM    MM    MM    MM  WmmmP"         MM         MM    8M     M8 MM 8M"""""" 8M        MM    
  M  `YM'   MM  8M   MM YM.    , MM    MM    MM    MM    MM YM.    ,       MM     ,M YM.    , 8M   MM    MM       MM    MM    MM    MM    MM 8M              MM         MM    YA.   ,A9 MM YM.    , YM.    ,  MM    
.JML. `'  .JMML.`Moo9^Yo.YMbmd'.JMML  JMML..JMML..JMML  JMML.`Mbmmd'     .JMMmmmmMMM  `Mbmmd' `Moo9^Yo..JMML.   .JMML  JMML..JMML..JMML  JMML.YMMMMMb      .JMML.     .JMML.   `Ybmd9'  MM  `Mbmmd'  YMbmd'   `Mbmo 
                                                                                                                                             6'     dP                               QO MP                          
                                                                                                                                             Ybmmmd'                                 `bmP                           
====================================================================================================================================================================================================================
/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
====================================================================================================================================================================================================================

====================================
**	Machine Learning Classifiers  **
====================================
This folder contains three machine learning classifiers implemented in Python: a convolutional neural network (CNN), k-nearest neighbors (KNN), and decision tree. 
Each classifier is located in its own subfolder and has its own .py file. 
The classifiers are trained on a dataset found in the `dataset` folder located in the root directory of this repository. 
The dataset folder contains two subfolders representing the two classes being classified: `occupied` and `unoccupied`.

It is important the directory structure provided be maintained so as not to introduce errors when obtaining files located in specific locations.
If you encounter import issues or similair, it may be due to the current working directory that is specified in the IDE is not pointed to the location. 

* Important *
The CWD should, if applicable, be set to ./MachineLearningClassifiers/

Virtual Environment
-------------------
The conda v-env .yaml file has been provided should the developer wish to import this environment directly into anaconda's navigator. It includes all the necessary requirement libraries.

====================
**  Requirements  **
====================
The following Python packages are required to run the classifiers:
The "requirements.txt" file specifies the exact version number for each library used.

keras
matplotlib
numpy
opencv_python
pandas
scikit_learn
scipy
seaborn
tensorflow
Pillow

=============
**  Usage  **
=============
Each classifier can be executed by running the "main.py" script from the command line, with two required arguments. 
To run the script, use the command "python main.py -c {} -m {}". The "-c" argument accepts a letter corresponding to the desired classifier (e.g., 'k' for K-Nearest Neighbor, 'd' for Decision Tree, or 'v' for VGG-16). 
The "-m" argument specifies the method of image loading: 'p' for loading and preprocessing images or 'o' for loading original images.

* NOTE *

The VGG-16 Classifier only accepts coloured images (4-dim shaped data), executing using the 'p' flag will cause an error to occur. Instead, use only the 'o' argument when running the VGG-16 Classifier.
It is suspected that there s an issue with the method in which the reduction of colour channels for the images is performed, reducing the shape to 3-Dim rather than 4-Dim with the inclusion of 1 for the colour channel dimension. 
This is an implementation that will be corrected in the future work of this project.

Each classifier uses the "dataset" directory to perform a k-fold cross-validation, splitting the data into training and testing sets.

==============
**  Output  **
==============
The classifiers save several output files in the subfolder where they are located:

A graph showing the performance metrics of the classifier. (ALL)
An average confusion matrix for the 5 folds that were run. (ALL)
An Excel spreadsheet containing the best parameters and their mean time when using GridSearchCV. (ALL)
A Scatter Graph overlaid with a Voronoi Diagram showing the features of the images in a feature space. (KNN)

===================
**  Classifiers  **
===================

Convolutional Neural Network (CNN) / VGG-16
-------------------------------------------
The CNN is implemented using TensorFlow and consists of several convolutional layers followed by fully connected layers. 
The CNN is located in the VGG-16Classifier subfolder and is contained in the VGG-16Classifier.py file.

K-Nearest Neighbor
------------------
The KNN classifier is implemented using scikit-learn and uses the k-nearest neighbors algorithm to classify the data. 
The KNN classifier is located in the KNNClassifier subfolder and is contained in the KNNClassifier.py file.

Decision Trees
--------------
The decision tree classifier is also implemented using scikit-learn and uses decision trees to classify the data. 
The decision tree classifier is located in the DecisionTreeClassifier subfolder and is contained in the DTreeClassifier.py file.

=============
**  Dataset  **
=============
The dataset used to train and test the classifiers is located in the dataset folder. 
The dataset consists of two subfolders: occupied and unoccupied. Each subfolder contains samples of the corresponding class.

=============
**  Author  **
=============
Aiden Parker
University Handle - 24631698

Date Last Modified
------------------
Date : 15/04/23