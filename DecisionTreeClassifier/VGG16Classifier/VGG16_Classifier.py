import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import (
    applications,
    layers,
    models,
    optimizers,
    callbacks,
    )
from pathlib import Path
from sklearn import (metrics, model_selection)


class VGG16Classifier:
    def __init__(
            self,
            X,
            y,
            flag,
            epochs=20,
            batch_size=8,
            image_width=224,
            image_height=224
    ):
        self.X = X
        self.y = y
        self.flag = flag

        self.current_dir = Path(__file__).parent
        self.weights_file_path: Path = self.current_dir / "best_performing_weights.h5"
        self.image_width, self.image_height = image_width, image_height
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.model = None
        if flag:
            self.text = "Pre-Processed Data"
        else:
            self.text = "Original Data"

        if tf.config.list_physical_devices('GPU'):
            self.device = 'GPU:0'
            print("Using GPU")
        else:
            self.device = 'CPU:0'
            print("Using CPU")

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
            cm_dir = Path(self.current_dir /
                          '{} - Confusion Matrices'.format(self.text))
            cm_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                cm_dir /
                f'VGG-16 - Confusion Matrix for Fold #{fold}.png')
            plt.show()
        else:
            ax.set_title('VGG-16 - Generalisability Confusion Matrix')
            cm_dir = Path(self.current_dir /
                          '{} - Generalisability Confusion Matrix'.format(self.text))
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

        barchart_dir = Path(self.current_dir /
                            '{} - Perforamnce Metrics Bar Charts'.format(self.text))
        barchart_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            barchart_dir /
            f'VGG-16 - Performance Metrics Bar Chart for Fold #{fold}.png')
        plt.show()

    def create_model(self):
        vgg_model = applications.VGG16(
            weights='imagenet', include_top=False, input_shape=(
                self.image_width, self.image_height, 3))

        for layer in vgg_model.layers:
            layer.trainable = False

        x = vgg_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        # x = layers.Dropout(0.5)(x) # Saw no perforamnce improvement with this
        # layer
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

    def main(self):
        """
        Description
        -----------
            Main entry point of the classifier.

        Handles the creation of the classifier, training, testing and performance evaluation
        """
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
        labels = ['Unoccupied', 'Occupied']
        folds = 5
        strat_kfold = model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=42)

        accuracies, precisions, recalls, f1s = [], [], [], []
        total_cm = np.zeros((len(labels), len(labels)))

        X_train_cv, X_test, y_train_cv, y_test = model_selection.train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        for fold, (train_idx, val_idx) in enumerate(
                strat_kfold.split(X_train_cv, y_train_cv), start=1):
            print(f"Training on fold {fold}")
            X_train, X_val = X_train_cv[train_idx], X_train_cv[val_idx]
            y_train, y_val = y_train_cv[train_idx], y_train_cv[val_idx]

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (X_train_cv, y_train_cv))
            train_dataset = train_dataset.map(self.preprocess_image).batch(
                self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.map(self.preprocess_image).batch(
                self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            test_dataset = test_dataset.map(self.preprocess_image).batch(
                self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

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

            with tf.device(self.device):
                history = self.run_classification(
                    train_dataset, val_dataset, model_callbacks)

                y_pred = (self.model.predict(test_dataset) > 0.5).astype(int)

            self.plot_accuracy_loss(history, fold)

            """================================================================
            CREATE A CONFUSION MATRIX BASED ON THE PREDICTIONS AND GROUND TRUTH
            ================================================================"""
            cm = self.create_conf_matrix(y_test, y_pred, labels, fold)
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
            print(metrics.classification_report(y_test, y_pred,
                  target_names=labels))

        """====================================================================
         * CREATE AN AVERAGE CONFUSION MATRIX BASED ON THE ONE FROM EACH FOLD *
        ===================================================================="""
        average_cm = total_cm / folds
        ax = sns.heatmap(average_cm, annot=True, fmt='.0f', annot_kws={
            'size': 10}, xticklabels=labels, yticklabels=labels, cmap='Blues')
        ax.set_title('Averaged VGG-16 Confusion Matrix Results')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')

        average_conf_matrix_dir = Path(
            self.current_dir / '{} - Averaged Confusion Matrices'.format(self.text))
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
            self.current_dir / '{} - Averaged Performance Metrics'.format(self.text))
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
            self.current_dir / '{} - Training and Validation Statistics'.format(self.text))
        model_stats_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            model_stats_dir /
            f'VGG-16 - Model Performance Statistics for Fold #{fold}.png')
        plt.show()

    # Function to predict the class of an image
    def predict(self, image_path):
        img = Image.open(image_path).resize((self.image_width, self.image_height))
        img = np.array(img)
        img = img / 255
        img = np.expand_dims(img, axis=0)

        with tf.device(self.device):
            prediction = self.model.predict(img)
            print('Prediction Score: ', prediction)
            prediction = np.argmax(prediction)

            if prediction == 0:
                print('Unoccupied')
            else:
                print('Occupied')