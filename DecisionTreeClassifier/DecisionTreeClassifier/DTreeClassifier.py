import cv2
import time
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import (model_selection, tree, metrics)


class DTreeClassifier:
    def __init__(self, X, y, flag, max_depth=None, seed=42):
        """
        Description
        -----------
            Initialise the instance of the class.

        This method initialises the Decision Tree Classifier class from `sklearn`,
        validates the current_dir and dataset_dir, and reads image files from the

        Parameters
        ----------
        X (array-like): Training image data.
        y (array-like): Training image labels.
        current_dir (str): Directory to save the output files.
        dataset_dir (str): Directory containing the dataset with subfolders.
        max_depth (int, optional): Maximum depth of the decision tree. Defaults to 12.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises
        ------
        ValueError: If the current_dir or dataset_dir does not exist.
        ValueError: If the dataset_dir does not contain any subfolders.

        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.current_dir = Path(__file__).parent
        self.max_depth = max_depth
        self.flag = flag
        if flag:
            self.text = "Pre-Processed Data"
        else:
            self.text = "Original Data"
        self.d_tree = tree.DecisionTreeClassifier(
            random_state=seed, max_depth=max_depth)

    def run_classification(self, X, y, X_true):
        """
        Description
        -----------
            Load images and labels from a given directory path.

        Parameters
        ----------
            X (array-like): Training image data.
            y (array-like): Training image labels.
            X_true (array-like): Test image data.

        Returns
        -------
            y_pred (array-like): Predicted labels for the test data.
        """
        print('Training on {} images.\nExecuting Train..'.format(len(X)))
        self.d_tree.fit(X, y)
        print('Training complete\n')

        print('Testing on {} images..'.format(len(X_true)))
        y_pred = self.d_tree.predict(X_true)
        print('Testing complete.')

        return y_pred

    def create_conf_matrix(self, y_true, y_pred, labels, fold):
        """
        Description
        -----------
            This function creates a confusion matrix based on the values provided for y_true and y_pred.
            It then uses this confusion matrix to create a heatmap. Both the confusion matrix and
            heatmap are returned from the function.

        Parameters
        ----------
          y_true (array-like): True labels for the testing set.
          y_pred (array-like): Predicted labels for the testing set.

        Returns
        -------
          cm (array-like): The confusion matrix.
          ax (matplotlib Axes): Axes object representing the heatmap plot of the confusion matrix.

        Raises
        ------
            ValueError: If y_true or y_pred is not provided a value
         """
        if not any(y_true) and any(y_pred):
            raise ValueError(
                f'ERROR: Expected `X` and `y` to contain at least 1 element, got X:({len(y_true)}), y:({len(y_pred)}).')

        cm = metrics.confusion_matrix(y_true, y_pred)
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='.0f',
            xticklabels=labels,
            yticklabels=labels,
            annot_kws={
                'size': 10},
            cmap='Blues')

        ax.set_title(f'Decision Tree Confusion Matrix for Fold #{fold}')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')

        cm_dir = Path(self.current_dir /
                      '{} - Confusion Matrices'.format(self.text))
        cm_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            cm_dir / f'Confusion Matrix for Fold #{fold}.png')
        plt.show()

        return cm

    def compute_accuracy(self, cm):
        """
        Description
        -----------
            Compute the accuracy value given a confusion matrix.

        Accuracy is a measure of a model's ability to correctly identify both true positives and true negatives.
        It is calculated as the ratio of the sum of true positives and true negatives to the sum of all elements
        in the confusion matrix.

        Parameters
        ----------
        cm (array-like 2-Dim): A 2-dimensional NumPy array representing the confusion matrix.

        Returns
        -------
        accuracy (float): The computed accuracy value.

        Raises
        ------
        TypeError: If the provided confusion matrix is not a 2-dimensional NumPy array.
        """
        if not (isinstance(cm, np.ndarray) and cm.ndim == 2):
            raise TypeError(f'Expected 2-Dim array-like, got {cm = }')
        tn = cm[0][0]
        fn = cm[0][1]
        tp = cm[1][1]
        fp = cm[1][0]
        accuracy = (tn + tp) / (tn + fp + tp + fn)
        return accuracy

    def compute_precision(self, cm):
        """
        Description
        -----------
            Compute the precision value given a confusion matrix.

        Precision is a measure of a model's ability to correctly identify true positives.
        It is calculated as the ratio of true positives to the sum of true positives and false positives.

        Parameters
        ----------
        cm (array-like 2-Dim): A 2-dimensional NumPy array representing the confusion matrix.

        Returns
        -------
        precision (float): The computed precision value.

        Raises
        ------
        TypeError: If the provided confusion matrix is not a 2-dimensional NumPy array.
        """
        if not (isinstance(cm, np.ndarray) and cm.ndim == 2):
            raise TypeError(f'Expected 2-Dim array-like, got {cm = }')
        tp = cm[1][1]
        fp = cm[1][0]
        precision = tp / (tp + fp)
        return precision

    def compute_recall(self, cm):
        """
        Description
        -----------
            Compute the recall value given a confusion matrix.

        Recall is a measure of a model's ability to correctly identify true positives.
        It is calculated as the ratio of true positives to the sum of true positives and false negatives.

        Parameters
        ----------
        cm (array-like 2-Dim): A 2-dimensional NumPy array representing the confusion matrix.

        Returns
        -------
        recall (float): The computed recall value.

        Raises
        ------
        TypeError: If the provided confusion matrix is not a 2-dimensional NumPy array.
        """
        if not (isinstance(cm, np.ndarray) and cm.ndim == 2):
            raise TypeError(f'Expected 2-Dim array-like, got {cm = }')

        fn = cm[0][1]
        tp = cm[1][1]
        recall = tp / (tp + fn)
        return recall

    def compute_f1(self, precision, recall):
        """
        Description
        -----------
            Compute the F1 score given precision and recall values.

        The F1 score is a measure of a model's accuracy, combining both precision and recall.
        It is calculated as the harmonic mean of precision and recall.

        Parameters
        ----------
        precision (float or int): The precision value of the model (true positives / (true positives + false positives)).
        recall (float or int): The recall value of the model (true positives / (true positives + false negatives)).

        Returns
        -------
        f1 (float): The computed F1 score.

        Raises
        ------
        TypeError
            If the provided precision or recall values are not of type float or int.
        """
        if not (
            isinstance(
                precision, (float, int)) and isinstance(
                recall, (float, int))):
            raise TypeError(
                f'Expected numerical value, got {precision = } {recall = } ')

        f1 = 2 * ((precision * recall) / (precision + recall))
        return f1

    def create_barchart(self, metric_labels, performance_metrics, fold):
        """
        Description
        -----------
            Create a bar chart to visualize the performance metrics of a decision tree model.

        This function generates a bar chart displaying the performance metrics for a decision tree
        model for a specific fold in a cross-validation process. The chart is saved as a PNG file
        in a specified directory.

        Parameters
        ----------
        metric_labels (array-like): A list of strings containing the labels for the performance metrics.
        performance_metrics (array-like): A list of floats containing the values of the performance metrics.
        fold (int): The index of the current fold in a cross-validation process.
        """
        plt.bar(metric_labels, performance_metrics)
        plt.title(f'Decision Tree Performance Metrics for Fold #{fold}')
        plt.xlabel('Performance Metrics')
        plt.ylabel('Metric Value')

        barchart_dir = Path(self.current_dir /
                            '{} - Perforamnce Metrics Bar Charts'.format(self.text))
        barchart_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            barchart_dir /
            f'Performance Metrics Bar Chart for Fold #{fold}.png')
        plt.show()

    def compute_avg_metrics(self, accuracies, precisions, recalls, f1s):
        """
        Description
        -----------
            Load images and labels from a given directory path.

        Parameters
        ----------
            accuracies (array-like): List of accuracy scores.
            precisions (array-like): List of precision scores.
            recalls (array-like): List of recall scores.
            f1s (array-like): List of F1 scores.

        Returns
        -------
            avg_metrics (Dictionary): Containing the mean values of each evaluation metric.

        Raises
        ------
            ValueError: If One or more of the performance metric arrays is empty.
        """
        if not (accuracies, precisions, recalls, f1s):
            raise ValueError(
                'One or more of the performance metric arrays was empty.')
        avg_metrics = {'Accuracy': np.mean(accuracies),
                       'Precision': np.mean(precisions),
                       'Recall': np.mean(recalls),
                       'F1': np.mean(f1s)}

        return avg_metrics

    def create_decision_tree_visualiser(self, class_names, fold):
        """
        Description
        -----------
            Display and save the decision tree visualisor.

        This method uses the trained decision tree classifier to create a visualisation
        of the tree structure, saves the visualisation as an image file, and displays
        the visualisation using `plot()`.

        Parameters
        ----------
        X (array-like): Training image data.
        y (array-like): Training image labels.
        class_names (list): List of class names for the labels.
        fold (int): Fold number for cross-validation (used in the saved image filename).
        """

        plt.figure(figsize=(80, 20))
        tree.plot_tree(
            self.d_tree,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10)

        tree_dir = Path(self.current_dir /
                        '{} - Tree Visualisations'.format(self.text))
        tree_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            tree_dir / f'Full Decision Tree Visualiser for Fold #{fold}.png')
        plt.show()

        plt.figure(figsize=(30, 20))
        # Create Cropped version of Tree Visualiser with depth of 3.
        tree.plot_tree(
            self.d_tree,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=12,
            max_depth=3)

        tree_dir = Path(self.current_dir /
                        '{} - Cropped Tree Visualisations'.format(self.text))
        tree_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            tree_dir /
            f'Cropped Decision Tree Visualiser for Fold #{fold}.png')
        plt.show()

    def show_misclassified_images(
            self,
            images,
            num_images,
            text,
            image_shape,
            fold):
        """
        Description
        -----------
            Display misclassified images.

        This method takes a list of misclassified images, resizes them, and displays
        them one by one with a window title indicating the fold number and misclassification type.

        Parameters
        ----------
        images (list): List of misclassified images.
        num_images (int): Number of images to display.
        text (str): Text describing the misclassification type (e.g., 'unoccupied' or 'occupied').
        image_shape (tuple): Tuple containing the original shape of the image (height, width, channels).

        """
        for i, image in enumerate(images[:num_images]):
            # Reshape the image back to its original shape
            image = image.reshape(image_shape)
            # Resize the image to 400x400 pixels for better visualization
            resized = cv2.resize(image, (400, 400))
            # Show the resized image in a separate window with a label
            cv2.imshow(
                f"Fold {fold} - Misclassified {text} Image {i}", resized)
            # Wait for a key event and close the window
            key = cv2.waitKey(0)
            # If ESC key is pressed, cease showing images.
            if key == 27:
                break
        cv2.destroyAllWindows()

    def create_best_param_dataframe(self, results):
        """
        Description
        -----------
            Create a DataFrame from the given results and save it as an Excel file.

        This method creates a DataFrame from the provided results dictionary, attempts to save
        the DataFrame as an Excel file in the specified save directory, and prints a filtered
        view of the results.

        Parameters
        ----------
        results (dict): Dictionary containing the results of the grid search.

        Raises
        ------
        OSError: If the file cannot be saved due to an OS error.
        PermissionError: If the file cannot be saved due to insufficient write permissions.

        """
        results_df = pd.DataFrame(results)
        print(
            f'Attempting to save Dataframe as Excel spreadsheet to `{self.current_dir}`..')

        dataframe_save_dir = Path(self.current_dir /
                                  '{} - GridSearchCV Data Frame'.format(self.text))
        dataframe_save_dir.mkdir(parents=True, exist_ok=True)
        try:
            results_df.to_excel(
                dataframe_save_dir /
                'dtree_classifier_gridsearchcv_results.xlsx')
            print('Dataframe Saved.')
        except (OSError, PermissionError):
            print(
                f'\nWARNING: Failed to save dataframe at location `{self.current_dir}`.')
            print('The file may be in use or write permissions are not enabled.\n')

        print('Filtered results after Hyper Parameter Tuning..')
        print(results_df[['param_max_depth', 'mean_test_score']])

    def find_best_params(self, min_depth=1, max_depth=14,):
        """
        Description
        -----------
            Find the optimal depth for the decision tree classifier using grid search.

        This method reshapes the input data and performs a grid search with cross-validation
        to find the optimal depth value for the decision tree classifier. It also creates
        a DataFrame of the results and prints the optimal depth and corresponding accuracy.

        This functions code was influenced based on the web page cited below,
        this code was not taken verbatim and was instead used to form the foundation.

        Author - Chetan Ambi
        Date - 30 September 2022
        Language - Python
        Website - PythonSimplified
        Web Page - How to use K-Fold CV and GridSearchCV with Sklearn Pipeline
        Web Address - https://pythonsimplified.com/how-to-use-k-fold-cv-and-gridsearchcv-with-sklearn-pipeline/

        Parameters
        ----------
        X (array-like): Training image data.
        y (array-like): Training image labels.
        min_depth (int, optional): Minimum value of `max_depth` to try. Defaults to 1.
        max_depth (int, optional): Maximum value of `max_depth` to try. Defaults to 14 for limited memory availability.

        Returns:
        optimal_depth (int): The optimal depth value for the decision tree classifier based on the dataset.

        """
        # Reshape X to a 2-Dim array.
        X = self.X.reshape(self.X.shape[0], -1)
        # Set the range of `max_depth` values to try.
        tree_depths = np.arange(min_depth, max_depth)

        cross_val = 5
        gridcv = model_selection.GridSearchCV(
            self.d_tree, cv=cross_val, param_grid={
                'max_depth': tree_depths}, scoring='accuracy')

        print('Finding the optimal `max_depth` value based on the dataset provided..')
        start = time.time()
        gridcv.fit(X, self.y)
        end = time.time()
        print('Optimal parameter found.')

        print(
            f'Time of computation for GridSearchCV on `max_depth` 1-14 : {end - start :.2f}s.')
        self.create_best_param_dataframe(results=gridcv.cv_results_)

        # Get the best accuracy for the best performing `max_depth` value.
        best_accuracy = gridcv.best_score_
        optimal_depth = gridcv.best_params_['max_depth']
        print(
            f'Best performing `max_depth` value = `{optimal_depth}` which had an accuracy rating of {best_accuracy*100:.2f}%.')

        return optimal_depth

    def main(self):
        """====================================================================
        *****************   MAIN ENTRY POINT TO THE PROGRAM   *****************
        ===================================================================="""
        if not (self.X.any() and self.y.any()):
            raise ValueError(
                f'ERROR: Expected `X` and `y` to contain at least 1 element, got X:({len(self.X)}), y:({len(self.y)}).')

        self.max_depth = self.find_best_params()
        if self.max_depth is None:
            print(
                'Best value for `max_depth` was not retrieved. Using default value `None`.')
            self.max_depth = 14

        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        labels = ['Unoccupied', 'Occupied']

        image_shape = tuple(self.X.shape[1:])

        """========================================================================
        ************  INITIALISE STRATIFIED K-FOLD CROSS VALIDATION   *************
        ========================================================================"""
        folds = 5
        strat_k_fold = model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=42)

        # Initialise the performance metrics lists to store each metric's
        # value.
        accuracies, precisions, recalls, f1s = [], [], [], []

        # Stores running total of confusion matrix results, computes average cm
        total_cm = np.zeros((len(labels), len(labels)))

        for fold, (train_idx, test_idx) in enumerate(
                strat_k_fold.split(self.X, self.y), start=1):
            print(f"\nFold: {fold}\n{'-'*15}")

            """====================================================================
            **   GENERATE THE X, y TRAIN AND TEST DATA SET (DONE FOR EACH FOLD)  **
            ===================================================================="""

            """  """
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_true, y_true = self.X[test_idx], self.y[test_idx]

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_true = X_true.reshape(X_true.shape[0], -1)

            """====================================================================
            **   PRINT THE NUMBER OF IMAGES IN EACH CLASS OF THE TESTING DATASET **
            ===================================================================="""

            # 0 = Unoccupied, 1 = Occupied
            num_unocc_true_images = len(X_true[y_true == 0])
            num_occ_true_images = len(X_true[y_true == 1])
            print(
                f'Number of Unoccupied Images in Test set : {num_unocc_true_images}')
            print(
                f'Number of Occupied Images in Test set : {num_occ_true_images}')

            """====================================================================
            ******   INITIALISE CLASSIFIER AND EXECUTE TRAINING AND TESTING  ******
            ===================================================================="""
            # Initialise the classifier so as to maintain fresh results.
            d_tree = DTreeClassifier(
                X=self.X,
                y=self.y,
                flag=self.flag,
                max_depth=self.max_depth)

            y_pred = d_tree.run_classification(X_train, y_train, X_true)

            print('Creating Visualisation Tree..')
            d_tree.create_decision_tree_visualiser(
                class_names=labels, fold=fold)
            print('Visualisation Tree Created.')

            """====================================================================
            *** RETRIEVE WHICH IMAGES WHICH WERE MISCLASSIFIED AND DISPLAY THEM ***
            ===================================================================="""

            # Retrieve only the indices and images of occupied images (1) that
            # were incorrectly labeled as unoccupied (0).
            occupied_images_indices = np.where(
                (y_pred == 0) & (y_true == 1))[0]
            misclassified_occupied_images = X_true[occupied_images_indices]

            # Show the first 20 images of the misclassified occupied class
            d_tree.show_misclassified_images(
                images=misclassified_occupied_images, num_images=20,
                text='Occupied', image_shape=image_shape, fold=fold)

            # Reset the array containing misclassified occupied images so as
            # not to re-show them during each fold.
            misclassified_occupied_images = np.array([])

            # Retrieve only the indices and images of unoccupied images (0)
            # that were incorrectly labeled as occupied (1).
            unoccupied_images_indices = np.where(
                (y_pred == 1) & (y_true == 0))[0]
            misclassified_unoccupied_images = X_true[unoccupied_images_indices]

            # Show the first 20 images of the misclassified unoccupied class
            d_tree.show_misclassified_images(
                images=misclassified_unoccupied_images, num_images=20,
                text='Unoccupied', image_shape=image_shape, fold=fold)

            # Reset the array containing misclassified unoccupied images so as
            # not to re-show them during each fold.
            misclassified_unoccupied_images = np.array([])

            """====================================================================
            * CREATE A CONFUSION MATRIX BASED ON THE PREDICTIONS AND GROUND TRUTH *
            ===================================================================="""
            cm = d_tree.create_conf_matrix(y_true, y_pred, labels, fold)
            total_cm += cm

            """====================================================================
            ******* CALCULATE THE PERFORMANCE METRICS FOR THE CURRENT FOLD  *******
            ===================================================================="""
            accuracy = self.compute_accuracy(cm)
            precision = self.compute_precision(cm)
            recall = self.compute_recall(cm)
            f1 = self.compute_f1(precision, recall)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            """====================================================================
            **** CREATE A BAR CHART SHOWING THE PERFORMANCE METRICS COMPUTED   ****
            ===================================================================="""
            performance_metrics = [accuracy, precision, recall, f1]
            d_tree.create_barchart(metric_labels, performance_metrics, fold)

            """====================================================================
            ***  CREATE A CLASSIFICATION REPORT SHOWING THE PERFORMANCE METRICS ***
            ===================================================================="""
            print(f'\nClassificaiton Report for Fold #{fold}\n', '-' * 20)
            print(metrics.classification_report(y_true, y_pred,
                  target_names=labels))

        """========================================================================
        ***  CREATE AN AVERAGE CONFUSION MATRIX BASED ON THE ONE FROM EACH FOLD ***
        ========================================================================"""
        average_cm = total_cm / folds
        ax = sns.heatmap(average_cm, annot=True, fmt='.0f', annot_kws={
            'size': 10}, xticklabels=labels, yticklabels=labels, cmap='Blues')
        ax.set_title('Averaged Decision Tree Confusion Matrix Results')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')

        average_conf_matrix_dir = Path(
            self.current_dir /
            '{} - Averaged Confusion Matrix'.format(self.text))
        average_conf_matrix_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            average_conf_matrix_dir / f'Averaged Confusion Matrix.png')
        plt.show()

        """========================================================================
        ** CREATE AN AVERAGE PERFORMANCE METRICS BAR CHART BASED ON THE RECORDED **
        ========================================================================"""
        # Calculate and print the average performance metrics across all folds
        avg_metrics = d_tree.compute_avg_metrics(
            accuracies, precisions, recalls, f1s)

        print(f'Average Performance Metrics - \n{avg_metrics}')
        plt.bar(avg_metrics.keys(), avg_metrics.values())
        plt.title('Decision Tree Averaged Performance Metrics')
        plt.bar(avg_metrics.keys(), avg_metrics.values())
        average_conf_matrix_dir = Path(
            self.current_dir /
            '{} - Averaged Performance Metrics'.format(self.text))
        average_conf_matrix_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            average_conf_matrix_dir / f'Averaged Performance Metrics.png')
        plt.show()
