import argparse
from pathlib import Path
from DecisionTreeClassifier.DTreeClassifier import DTreeClassifier
from KNNClassifier.KNNClassifier import KNNClassifier
from HelperFiles.HelperClass import HelperClass
from VGG16Classifier.VGG16_Classifier import VGG16Classifier


def get_classifier(arg, X, y, flag):
    """Returns an instance of a classifier based on the provided argument.

    Parameters
    ----------
        arg (str): The classifier argument (v for VGG16Classifier, k for KNNClassifier, or d for DTreeClassifier).
        X (numpy.ndarray): The feature matrix.
        y (numpy.ndarray): The target vector.
        flag (bool): A flag to indicate whether to print debug messages.

    Returns
    -------
        Classifier: An instance of the specified classifier.

    Raises
    ------
        ValueError: If the provided argument is invalid.
    """
    if arg == 'v':
        classifier = VGG16Classifier(
            X=X,
            y=y,
            epochs=20,
            image_width=IMG_WIDTH,
            image_height=IMG_HEIGHT,
            flag=flag)
        classifier.create_model()
        return classifier
    elif arg == 'k':
        return KNNClassifier(X=X, y=y, flag=flag)
    elif arg == 'd':
        return DTreeClassifier(X=X, y=y, flag=flag)
    else:
        raise ValueError("Invalid classifier argument")


def parse_arguments():
    """Parses command-line arguments for the machine learning image classifiers.

    Returns
    -------
        Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Machine Learning Image Classifiers.")
    parser.add_argument(
        "-c",
        "--classifier",
        choices=[
            'v',
            'k',
            'd'],
        required=True,
        help="Choose Classifier: 'v' for using VGG-16, 'd' for using Decision Tree, or 'k' for using K-Nearest Neighbor.")
    parser.add_argument(
        "-m",
        "--mode",
        choices=[
            'o',
            'p'],
        help="Choose mode: 'o' for loading images with no pre-processing, or 'p' for loading and pre-processing the images.")
    return parser.parse_args()


# Main Entry point of the program
if __name__ == '__main__':
    CURRENT_DIR = Path(__file__).parent
    DATASET_DIR = CURRENT_DIR / "dataset"
    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    if not DATASET_DIR.exists():
        raise ValueError(
            f"The dataset directory `{DATASET_DIR}` does not exist.")

    args = parse_arguments()

    if not args.mode:
        raise ValueError(
            "Image loading mode is required. See --help for info.")

    helper = HelperClass()
    if args.mode == 'o':
        (X, y) = helper.load_files_from_dir_supervised(
            DATASET_DIR, IMG_WIDTH, IMG_HEIGHT)
        flag = 0
    elif args.mode == 'p':
        (X, y) = helper.preprocess_files_from_dir(DATASET_DIR)
        flag = 1

    classifier = get_classifier(
        arg=args.classifier,
        X=X,
        y=y,
        flag=flag)
    
    classifier.main()
    