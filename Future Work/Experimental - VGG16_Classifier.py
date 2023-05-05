import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import (metrics, model_selection)
import tensorflow as tf
from tensorflow.keras import (
    applications,
    layers,
    models,
    optimizers,
    callbacks, 
    utils)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from HelperFiles.HelperClass import HelperClass
from collections import Counter
import cv2
import os

class DataGenerator(utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, image_shape, shuffle=True, augmentation=None):
        self.image_paths = image_paths
        self.labels = labels 
        self.batch_size = batch_size 
        self.image_shape = image_shape 
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes) 

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size)) # number of batches

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[i] for i in batch_indexes] # image paths
        batch_labels = [self.labels[i] for i in batch_indexes] # one-hot encoded labels
        X = np.empty((self.batch_size, *self.image_shape)) # unpack self.image_shape
        y = np.empty((self.batch_size), dtype=int) 

        for i, image_path in enumerate(batch_image_paths):
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.image_shape[:2])
            if self.augmentation:
                image = self.augmentation(image=image)["image"]
            X[i,] = image
            y[i] = batch_labels[i]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class VGG16Classifier:
    def __init__(
            self,
            save_dir,
            epochs=20,
            batch_size=8,
            weights_file_path='',
            image_width=224,
            image_height=224):

        self.save_dir: Path = Path(save_dir)

        if not self.save_dir.exists():
            raise ValueError(
                f'The save directory provided `{save_dir}` does not exist.')

        self.weights_file_path: Path = Path(weights_file_path)
        self.image_width, self.image_height = image_width, image_height
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.model = None

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

    def create_conf_matrix(self, y_true, y_pred, labels, fold=None):
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

        ax.set_title(f'VGG-16 Confusion Matrix for Fold #{fold}')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')
        
        if fold is not None:
            ax.set_title(f'VGG-16 Confusion Matrix for Fold #{fold}')
            cm_dir = Path(self.save_dir / 'Confusion Matrices')
            cm_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(cm_dir / f'VGG-16 - Confusion Matrix for Fold #{fold}.png')
            plt.show()
        else:
            ax.set_title('VGG-16 - Generalisability Confusion Matrix')
            cm_dir = Path(self.save_dir / 'Generalisability Confusion Matrix')
            cm_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(cm_dir / 'VGG-16 - Generalisable Confusion Matrix.png')
            plt.show()
        
        return cm

    def create_barchart(self, metric_labels, performance_metrics, fold):
        """
        Description
        -----------
            Create a bar chart to visualize the performance metrics of a VGG-16 model.

        This function generates a bar chart displaying the performance metrics for a VGG-16
        model for a specific fold in a cross-validation process. The chart is saved as a PNG file
        in a specified directory.

        Parameters
        ----------
        metric_labels (array-like): A list of strings containing the labels for the performance metrics.
        performance_metrics (array-like): A list of floats containing the values of the performance metrics.
        fold (int): The index of the current fold in a cross-validation process.
        """
        plt.bar(metric_labels, performance_metrics)
        plt.title(f'VGG-16 Performance Metrics for Fold #{fold}')
        plt.xlabel('Performance Metrics')
        plt.ylabel('Metric Value')

        barchart_dir = Path(self.save_dir / 'Perforamnce Metrics Bar Charts')
        barchart_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            barchart_dir / f'VGG-16 - Performance Metrics Bar Chart for Fold #{fold}.png')
        plt.show()

    def create_model(self):
        vgg_model = applications.VGG16(
            weights='imagenet', include_top=False, input_shape=(
                self.image_width, self.image_height, 3))

        for layer in vgg_model.layers:
            layer.trainable = False

        x = vgg_model.output
        x = layers.GlobalAveragePooling2D()(x)
        # x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=vgg_model.input, outputs=predictions)

        if self.weights_file_path.exists():
            model.load_weights(self.weights_file_path)
            print('Model Weights Loaded Successfully.')

        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        model.summary()
        self.model = model

    def run_classification(self, train_generator, val_generator, callbacks):
        print('Training the model..\n')
        history = self.model.fit(train_generator, epochs=self.epochs,
                                 validation_data=val_generator,
                                 callbacks=callbacks)
        print('\tModel Trained.')
        return history

    def preprocess_image(self, image, label):
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image, label

    def main(self, X, y, batch_size=32, n_splits=5):
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        labels = ['Unoccupied', 'Occupied']

        strat_kfold = model_selection.StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42)

        accuracies, precisions, recalls, f1s = [], [], [], []
        total_cm = np.zeros((len(labels), len(labels)))

        X_crossval, X_test, y_crossval, y_test = model_selection.train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        for fold, (train_idx, val_idx) in enumerate(
                strat_kfold.split(X_crossval, y_crossval), start=1):
            print(f"Training on fold {fold}")
            X_train, X_val = X_crossval[train_idx], X_crossval[val_idx]
            y_train, y_val = y_crossval[train_idx], y_crossval[val_idx]

            image_shape = (self.image_width, self.image_height, 3)
            train_generator = DataGenerator(X_train, y_train, batch_size, image_shape)
            val_generator = DataGenerator(X_val, y_val, batch_size, image_shape)
            test_generator = DataGenerator(X_test, y_test, batch_size, image_shape)

            """================================================================
            **   PRINT THE NUMBER OF IMAGES IN EACH CLASS OF THE TESTING DATASET **
            ================================================================"""

            num_unocc_train_images = len(X_train[y_train == 0])
            num_occ_train_images = len(X_train[y_train == 1])
            print(
                f'Number of Unoccupied Images in Train set : {num_unocc_train_images}')
            print(
                f'Number of Occupied Images in Train set : {num_occ_train_images}')

            # 0 = Unoccupied, 1 = Occupied
            num_unocc_val_images = len(X_val[y_val == 0])
            num_occ_val_images = len(X_val[y_val == 1])
            print(
                f'Number of Unoccupied Images in Validation set : {num_unocc_val_images}')
            print(
                f'Number of Occupied Images in Validation set : {num_occ_val_images}')

            num_unocc_test_images = len(X_test[y_test == 0])
            num_occ_test_images = len(X_test[y_test == 1])
            print(
                f'Number of Unoccupied Images in Test set : {num_unocc_test_images}')
            print(
                f'Number of Occupied Images in Test set : {num_occ_test_images}')

            """================================================================
            ****   INITIALISE CLASSIFIER AND EXECUTE TRAINING AND TESTING  ****
            ================================================================"""

            checkpoint = callbacks.ModelCheckpoint(
                filepath=self.weights_file_path,
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max',
            )
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True)
            model_callbacks = [checkpoint, early_stopping,]

            with tf.device('device:GPU:0'):
                history = self.run_classification(
                    train_generator, val_generator, model_callbacks)

                y_pred = (self.model.predict(test_generator) > 0.5).astype(int)

            self.plot_accuracy_loss(history, fold)

            """================================================================
            CREATE A CONFUSION MATRIX BASED ON THE PREDICTIONS AND GROUND TRUTH
            ================================================================"""
            y_true = []
            for _, batch_labels in test_generator:
                y_true.extend(batch_labels)
            y_true = np.array(y_true)
            
            cm = self.create_conf_matrix(y_true, y_pred, labels, fold)
            total_cm += cm

            """================================================================
            ***** CALCULATE THE PERFORMANCE METRICS FOR THE CURRENT FOLD  *****
            ================================================================"""
            accuracy = self.compute_accuracy(cm)
            precision = self.compute_precision(cm)
            recall = self.compute_recall(cm)
            f1 = self.compute_f1(precision, recall)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            """================================================================
            *** CREATE A BAR CHART SHOWING THE PERFORMANCE METRICS COMPUTED ***
            ================================================================"""
            performance_metrics = [accuracy, precision, recall, f1]
            self.create_barchart(metric_labels, performance_metrics, fold)

            """================================================================
            *  CREATE A CLASSIFICATION REPORT SHOWING THE PERFORMANCE METRICS *
            ================================================================"""
            print(f'\nClassificaiton Report for Fold #{fold}\n', '-' * 20)
            print(metrics.classification_report(y_true, y_pred,
                  target_names=labels))

        """====================================================================
         * CREATE AN AVERAGE CONFUSION MATRIX BASED ON THE ONE FROM EACH FOLD *
        ===================================================================="""
        average_cm = total_cm / n_splits
        ax = sns.heatmap(average_cm, annot=True, fmt='.0f', annot_kws={
            'size': 10}, xticklabels=labels, yticklabels=labels, cmap='Blues')
        ax.set_title('Averaged VGG-16 Confusion Matrix Results')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')

        average_conf_matrix_dir = Path(
            self.save_dir / 'Averaged Confusion Matrices')
        average_conf_matrix_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(average_conf_matrix_dir /
                    'VGG-16 - Averaged Confusion Matrix.png')
        plt.show()

        """====================================================================
        CREATE AN AVERAGE PERFORMANCE METRICS BAR CHART BASED ON THE RECORDED 
        ===================================================================="""
        # Calculate and print the average performance metrics across all folds
        avg_metrics = self.compute_avg_metrics(
            accuracies, precisions, recalls, f1s)

        print(f'Average Performance Metrics - \n{avg_metrics}')
        plt.bar(avg_metrics.keys(), avg_metrics.values())
        plt.title('VGG-16 Averaged Performance Metrics')

        average_barchart_dir = Path(
            self.save_dir / 'Averaged Performance Metrics')
        average_barchart_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(average_barchart_dir /
                    'VGG-16 - Averaged Performance Metrics Bar Chart.png')
        plt.show()

    def plot_accuracy_loss(self, history, fold):
        plt.plot(history.history['accuracy'], label='Training accuracy')
        plt.plot(history.history['val_accuracy'],
                 label='Validation accuracy')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Statistics - Fold #{fold}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        model_stats_dir = Path(
            self.save_dir / 'Training and Validation Statistics')
        model_stats_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            model_stats_dir / f'VGG-16 - Model Performance Statistics for Fold #{fold}.png')
        plt.show()

    def predict_new_images(self, X_test, y_test):
        if self.model is None:
            print(
                'Model has not been loaded. Create the model and load the necessary weights file.')

        # preprocessed_images = [map(self.preprocess_image, image) for image, label in zip(images, labels)]

        preprocessed_images = []
        for image, label in zip(X_test, y_test):
            processed_img, label = self.preprocess_image(image, label)
            preprocessed_images.append(processed_img)
        preprocessed_images = np.stack(preprocessed_images)
        
        print('Predicting the class for the given feature vectors..')
        predictions = self.model.predict(preprocessed_images)

        binary_preds = (predictions >= 0.5).astype(int).flatten()
        
        self.create_conf_matrix(y_test, binary_preds, labels=['Unoccupied', 'Occupied'], fold=None)
        
        true_preds = Counter()
        for truth, pred in zip(y_test, binary_preds):
            if truth == pred:
                true_preds[f'Correct Predicion for class_{truth}'] += 1
            else:
                true_preds[f'Inorrect Predicion for class_{truth}'] += 1

        return true_preds




    def load_files_from_dir_supervised(self, directory, width=128, height=128):
        subfolders = [f.lower() for f in os.listdir(directory)
                      if os.path.isdir(os.path.join(directory, f))]
        image_paths, labels = [], []
        for folder in subfolders:
            label = 1 if folder == 'occupied' else 0
            folder_path = os.path.join(directory, folder)
            for filename in os.listdir(folder_path):
                labels.append(label)
                image_paths.append(os.path.join(folder_path, filename))
    
        return np.array(image_paths), np.array(labels)


if __name__ == "__main__":
    """====================================================================
    *****************   MAIN ENTRY POINT TO THE PROGRAM   *****************
    ===================================================================="""
    # Specify the current working dir to save data to such as dataframes,
    # figures etc.
    current_dir: str = Path.cwd()
    dataset_dir: str = current_dir.parent / 'dataset'
    weights_file: str = current_dir / 'best_performing_weights.h5'
    img_width: int = 128
    img_height: int = 128
    
    classifier = VGG16Classifier(
        current_dir,
        epochs=20,
        image_width=img_width,
        image_height=img_height,
        weights_file_path=weights_file)
    classifier.create_model()
    
    X, y = classifier.load_files_from_dir_supervised(
        dataset_dir, width=img_width, height=img_height)


    classifier.main(X, y, batch_size=16)



    
    test_dataset = Path(current_dir / 'test_data')
    X_test, y_test = HelperClass().load_files_from_dir_supervised(
        test_dataset, width=img_width, height=img_height)
    
    
    predictions = classifier.predict_new_images(X_test, y_test)
    print('\n\t* Predictions *')
    for key, val in predictions.items():
        if key[-1] == '1':
            text = '(Occupied)'
        else:
            text = '(Unoccupied)'
        print(f'{key} {text} : {val}')
