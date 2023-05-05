import cv2
import time
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import (pyplot as plt, colors)
from scipy.spatial import (Voronoi, voronoi_plot_2d)
from sklearn import (model_selection, metrics, decomposition, neighbors)

class KNNClassifier():
    def __init__(self, X, y, flag, n_neighbors=1) -> None:
        """
        Description
        -----------
            Initialise the instance of the class.

        This method initialises the K-Nearest Neighbor (KNN) Classifier class from `sklearn`,
        validates the current_dir and dataset_dir, and reads image files from the

        Parameters
        ----------
        X (array-like): Training image data.
        y (array-like): Training image labels.
        current_dir (str): Directory to save the output files.
        dataset_dir (str): Directory containing the dataset with subfolders.
        n_neighbors (int, optional): Maximum number of neighbors in KNN classifier. Defaults to 1.

        Raises
        ------
        ValueError: If the current_dir or dataset_dir does not exist.
        ValueError: If the dataset_dir does not contain any subfolders.

        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.current_dir = Path(__file__).parent
        self.n_neighbors = n_neighbors
        self.flag = flag
        if flag:
            self.text = "Pre-Processed Data"
        else:
            self.text = "Original Data"
        self.knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

    def run_classification(self, X_train, y_train, X_true):
        """
        Description
        -----------
            Load images and labels from a given directory path.

        Parameters
        ----------
            X_train (list or array-like): Training image data.
            y_train (list or array-like): Training image labels.
            X_true (list or array-like): Test image data.

        Returns
        -------
            y_pred (list or array-like): Predicted labels for the test data.
        """
        print('Training on {} images.\nExecuting Train..'.format(len(X_train)))
        # Train the classifier on the training data
        self.knn.fit(X_train, y_train)
        print('Training complete\n')

        print('Testing on {} images'.format(len(X_true)))
        # Test the classifier on the testing data
        y_pred = self.knn.predict(X_true)
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
          y_true (list or array-like): True labels for the testing set.
          y_pred (list or array-like): Predicted labels for the testing set.

        Returns
        -------
          cm (array-like): The confusion matrix.
          ax (matplotlib Axes): Axes object representing the heatmap plot of the confusion matrix.

        Raises
        ------
            ValueError: If y_true or y_pred is not provided a value
         """
        if not (any(y_true) or any(y_pred)):
            raise ValueError(
                'No value for either `y_true` or `y_pred` was provided.')

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

        ax.set_title(
            f'{self.n_neighbors}-Nearest Neighbor Confusion Matrix for Fold #{fold}')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')

        cm_dir = Path(
            self.current_dir /
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
            Create a bar chart to visualize the performance metrics of a k-nearest neighbor model.

        This function generates a bar chart displaying the performance metrics for a k-nearest neighbor
        model for a specific fold in a cross-validation process. The chart is saved as a PNG file
        in a specified directory.

        Parameters
        ----------
        metric_labels (array-like): A list of strings containing the labels for the performance metrics.
        performance_metrics (array-like): A list of floats containing the values of the performance metrics.
        fold (int): The index of the current fold in a cross-validation process.
        """
        plt.bar(metric_labels, performance_metrics)
        plt.title(f'k-Nearest Neighbor Performance Metrics for Fold #{fold}')
        plt.xlabel('Performance Metrics')
        plt.ylabel('Metric Value')

        barchart_dir = Path(self.current_dir /
                            '{} - Perforamnce Metrics Bar Charts'.format(self.text))
        barchart_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            barchart_dir /
            f'Performance Metrics Bar Chart for Fold #{fold}.png')
        plt.show()

    def create_feature_space_graph(self, X_true, y_true, y_pred, labels, fold):
        """
        Description
        -----------
        Create a scatter plot of the feature space for image data, where each data point represents
        an image and its location is determined by a Principal Component Analysis (PCA) with target
        dimensions of 2. The plot shows the occupied and unoccupied rows of the PCA data in different
        colours based on the true and predicted labels, with a legend indicating the meaning of the colours.
        Voronoi tessellation is also added to the scatter plot.

        Scatter graph is shown after function is invoked.

        Args:
            X_test (array): the input image data for which to make predictions.
            y_test (array): true labels for each data point in X_test.
            knn (KNeighborsClassifier): the KNN classifier used to make predictions on X_test.
            fold (int): the fold number of K-Fold Cross Validation for indicative labelling of the current fold.

            https://realpython.com/visualizing-python-plt-scatter/#getting-started-with-pltscatter
        """
        cmap = colors.ListedColormap(['red', 'green'])

        """
        The below Principal Component Analysis finds a lower dimensional
        representation of the data whilst preserving the quality.
        Important use for the scatter graph of the data points in the
        feature space of the image data.
        """
        pca = decomposition.PCA(n_components=2)  # Target dimensions is 2.
        # Apply the same transformation to X_true
        X_true_pca = pca.fit_transform(X_true)

        # Extract the occupied and unoccupied rows from column 0 and 1 of the
        # PCA data.
        occ_rows = X_true_pca[:, 1]
        unocc_rows = X_true_pca[:, 0]

        vor = Voronoi(X_true_pca)

        # Plot the Voronoi diagram.
        fig, ax = plt.subplots(figsize=(8, 8))
        voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                        line_colors='gray', line_width=0.5)

        # Plot the data points for each image in the feature space using a
        # scatter plot.
        plt.scatter(unocc_rows, occ_rows, c=y_true, cmap=cmap, alpha=0.5)
        plt.scatter(unocc_rows, occ_rows, c=y_pred, cmap=cmap, marker='x')
        plt.title(
            f'Voronoi Tesselation Diagram for True and Predicted Images in the Feature Space for Fold #{fold}')

        # Add a legend to indicate the meaning of the colours used in the
        # scatter plot.
        class_colours = [cmap(i) for i in range(len(labels))]
        recs = [plt.Rectangle((0, 0), 1, 1, fc=class_colours[i])
                for i in range(len(labels))]
        plt.legend(recs, labels, loc='lower right')

        scatter_dir = Path(
            self.current_dir /
            '{} - Scatter Plots_Voronoi Diagrams'.format(self.text))
        scatter_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            scatter_dir /
            f'Feature Space Voronoi Diagram for Fold #{fold}.png')
        plt.show()

    def compute_avg_metrics(self, accuracies, precisions, recalls, f1s):
        """
        Description
        -----------
            Load images and labels from a given directory path.

        Parameters
        ----------
            accuracies (list or array-like): List of accuracy scores.
            precisions (list or array-like): List of precision scores.
            recalls (list or array-like): List of recall scores.
            f1s (list or array-like): List of F1 scores.

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

    def show_misclassified_images(
            self,
            images,
            num_images,
            text,
            image_shape,
            fold):
        """
        Display misclassified images in a separate window.

        This function shows the first `num_images` misclassified images in the `images` list.
        Images are reshaped to their original shape and resized for better visualisation.
        Images are displayed in separate windows with labels indicating the fold number,
        misclassification type, and image index.

        Parameters
        ----------
        images (array-like): A list or NumPy array containing the misclassified images.
        num_images (int): The number of misclassified images to display.
        text (str): Text description of the misclassification type (e.g., "Occupied" or "Unoccupied").
        image_shape (tuple): A tuple containing the original shape of the images (e.g., (128, 128, 3)).
        fold (int): The index of the current fold in a cross-validation process.
        """
        # Display the first `num_images` misclassified images
        for i, image in enumerate(images[:num_images]):
            image = image.reshape(image_shape)
            resized = cv2.resize(image, (400, 400))
            cv2.imshow(
                f"Fold {fold} - Misclassified {text} Image {i}", resized)
            key = cv2.waitKey(0)
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
            f'\nAttempting to save Dataframe as Excel spreadsheet to `{self.current_dir}`..')

        dataframe_save_dir = Path(self.current_dir /
                                  '{} - GridSearchCV Data Frame'.format(self.text))
        dataframe_save_dir.mkdir(parents=True, exist_ok=True)
        try:
            results_df.to_excel(
                dataframe_save_dir /
                'knn_classifier_gridsearchcv_results.xlsx')
            print('Dataframe Saved.')
        except (OSError, PermissionError):
            print(
                f'\nWARNING: Failed to save dataframe at location `{self.current_dir}`.')
            print('The file may be in use or write permissions are not enabled.\n')
        print('\nFiltered results after Hyper Parameter Tuning..')
        print(results_df[['param_n_neighbors', 'mean_test_score']])

    def find_best_params(self, min_neighbors=1, max_neighbors=13):
        """
        Description
        -----------
            Find the optimal n_neighbor for the K-Nearest Neighbor (KNN) using grid search.

        This method reshapes the input data and performs a grid search with cross-validation
        to find the optimal n_neighbor value for the KNN classifier. It also creates
        a DataFrame of the results and prints the optimal neighbors and corresponding accuracy.

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
        min_neighbors (int, optional): Minimum value of `n_neighbors` to try. Defaults to 1.
        max_neighbors (int, optional): Maximum value of `n_neighbors` to try. Defaults to 13 for limited memory availability.

        Returns:
        best_n_neighbor (int): The optimal `n_neighbors` value for the k-nearest neighbor classifier based on the dataset.

        """
        # Reshape `X` to a 2-Dim array.
        X = self.X.reshape(self.X.shape[0], -1)
        # Set the range of `n_neighbors` values to try.
        range_n_neighbors = np.arange(min_neighbors, max_neighbors)

        cross_val = 5
        gridcv = model_selection.GridSearchCV(
            self.knn, cv=cross_val, param_grid={
                'n_neighbors': range_n_neighbors}, scoring='accuracy')
        print('Finding the optimal `n_neighbors` value based on the dataset provided..')
        start = time.time()
        gridcv.fit(X, self.y)
        end = time.time()
        print(
            f'Time of execution for GridSearchCV using {cross_val}-folds and \
              {min_neighbors}-{max_neighbors} `n_neighbors` param = {end-start:.2f}s.')
        print('Optimal parameter found.')

        # Create a datafram of the results from the gridsearchcv and store it as
        # .xlsx file in `self.current_dir` location.
        self.create_best_param_dataframe(results=gridcv.cv_results_)

        # Get the best accuracy for the best performing `n_neighbors` value.
        best_accuracy = gridcv.best_score_
        best_n_neighbor = gridcv.best_params_['n_neighbors']
        print(
            f'Best `n_neighbors` value = `{best_n_neighbor}` which had an accuracy rating of {best_accuracy*100:.2f}%.')

        return best_n_neighbor

    def main(self):
        """====================================================================
        *****************   MAIN ENTRY POINT TO THE PROGRAM   *****************
        ===================================================================="""

        self.n_neighbors = self.find_best_params()
        if self.n_neighbors is None:
            print('Best value for `n_neighbors` was not retrieved. Using default (1).')
            self.n_neighbors = 1

        # Labels for the performance metrics bar chart.
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        # Labels for the confusion matrix.
        labels = ['Unoccupied', 'Occupied']
        # Check X, y contain data.

        # check if X and y arrays contain at least 1 element.
        if not (self.X.any() and self.y.any()):
            raise ValueError(
                f'ERROR: Expected `X` and `y` to contain at least 1 element, got X:({len(self.X)}), y:({len(self.y)}).')

        # Get the shape of the images, used for showing any misclassified
        # images since X_train and X_true are reshaped.
        image_shape = tuple(self.X.shape[1:])

        """========================================================================
        ************  INITIALISE STRATIFIED K-FOLD CROSS VALIDATION   *************
        ========================================================================"""
        # Set the amount of samples to split the training data into.
        folds = 5

        # Initialise the Stratified K-fold data splitter.
        strat_k_fold = model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=42)

        # Initialise the performance metrics lists to store each metric's value
        # per fold.
        accuracies, precisions, recalls, f1s = [], [], [], []

        # Initialise a 2-Dim array for the confusion matrix of each fold array
        # to compute the average classification rating.
        total_cm = np.zeros((len(labels), len(labels)))

        for fold, (train_idx, test_idx) in enumerate(
                strat_k_fold.split(self.X, self.y), start=1):
            print(f"\nFold: {fold}\n{'-'*15}")

            """====================================================================
            **   GENERATE THE X, y TRAIN AND TEST DATA SET (DONE FOR EACH FOLD)  **
            ===================================================================="""

            """  """
            # Split the data into training and test sets for the current fold.
            # X = image data, y = Image labels
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
            knn = KNNClassifier(
                X=self.X,
                y=self.y,
                flag=self.flag,
                n_neighbors=self.n_neighbors)

            # Train the classifier on the training data and get predictions for
            # the test set.
            y_pred = knn.run_classification(X_train, y_train, X_true)

            """====================================================================
            *** RETRIEVE WHICH IMAGES WHICH WERE MISCLASSIFIED AND DISPLAY THEM ***
            ===================================================================="""

            # Retrieve only the indices and images of occupied images (1) that
            # were incorrectly labeled as unoccupied (0).
            occupied_images_indices = np.where(
                (y_pred == 0) & (y_true == 1))[0]
            misclassified_occupied_images = X_true[occupied_images_indices]
            print(
                f'Number of Misclassified Occupied - {len(occupied_images_indices)}')

            # # Show the first 20 images of the misclassified occupied class
            knn.show_misclassified_images(
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
            print(
                f'Number of Misclassified Unoccupied - {len(unoccupied_images_indices)}')

            # # Show the first 20 images of the misclassified unoccupied class
            knn.show_misclassified_images(
                images=misclassified_unoccupied_images, num_images=20,
                text='Unoccupied', image_shape=image_shape, fold=fold)

            # Reset the array containing misclassified unoccupied images so as
            # not to re-show them during each fold.
            misclassified_unoccupied_images = np.array([])

            """================================================================
            *************    CREATE AND SAVE CONFUSION MATRIX    **************
            ================================================================"""
            # Create the confusion matrix and add it to the running total.
            cm = knn.create_conf_matrix(y_true, y_pred, labels, fold)
            total_cm += cm

            """====================================================================
            ******* CALCULATE THE PERFORMANCE METRICS FOR THE CURRENT FOLD  *******
            ===================================================================="""
            # Calucate each performance metric.
            accuracy = self.compute_accuracy(cm)
            precision = self.compute_precision(cm)
            recall = self.compute_recall(cm)
            f1 = self.compute_f1(precision, recall)

            # Add the performance metrics to the respective list.
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            """====================================================================
            **** CREATE A BAR CHART SHOWING THE PERFORMANCE METRICS COMPUTED   ****
            ===================================================================="""
            # Create a bar chart showing the performance metrics calculated and
            # save it.
            performance_metrics = [accuracy, precision, recall, f1]
            knn.create_barchart(metric_labels, performance_metrics, fold)

            """====================================================================
            ***  CREATE A CLASSIFICATION REPORT SHOWING THE PERFORMANCE METRICS ***
            ===================================================================="""

            print('\nClassificaiton Report\n', '-' * 20)
            print(metrics.classification_report(y_true, y_pred,
                  target_names=labels))

            """====================================================================
            CREATE A SCATTER GRAPH WITH OVERLAID VORONOI DIAGRAM TO VISUALISE FEATURES
            ===================================================================="""
            print('Creating Scatter Graph..')
            knn.create_feature_space_graph(
                X_true=X_true,
                y_true=y_true,
                y_pred=y_pred,
                labels=labels,
                fold=fold)
            print('Scatter Graph Created.')

        """========================================================================
        ***  CREATE AN AVERAGE CONFUSION MATRIX BASED ON THE ONE FROM EACH FOLD ***
        ========================================================================"""
        # Create an average confusion matrix for each confusion matrix computed
        # for each fold.
        average_cm = total_cm / folds
        ax = sns.heatmap(average_cm, annot=True, fmt='.0f', annot_kws={
            'size': 10}, xticklabels=labels, yticklabels=labels, cmap='Blues')
        ax.set_title('Averaged K-Nearest Neighbor Confusion Matrix Results')
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
        avg_metrics = knn.compute_avg_metrics(
            accuracies, precisions, recalls, f1s)
        print(f'Average Performance Metrics - \n{avg_metrics}')
        # Create a bar chart showing the performance metrics calculated
        plt.bar(avg_metrics.keys(), avg_metrics.values())
        plt.title('K-Nearest Neighbor Averaged Performance Metrics')
        average_conf_matrix_dir = Path(
            self.current_dir /
            '{} - Averaged Performance Metrics'.format(self.text))
        average_conf_matrix_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            average_conf_matrix_dir / f'Averaged Performance Metrics.png')
        plt.show()
