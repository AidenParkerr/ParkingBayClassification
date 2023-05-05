import cv2


class ImagePreProcessing():

    def full_preprocess(self, image, kernel=(3, 3), ksize=5):
        grey = self.convert_to_grey(image)
        gaus_blur = self.gaussian_blur(grey, kernel)
        median_blur = self.median_blur(gaus_blur, ksize)
        return median_blur

    def resize(self, image, size=(128, 128)):
        return cv2.resize(image, size)

    def convert_to_grey(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image, kernel=(3, 3)):
        return cv2.GaussianBlur(image, kernel, sigmaX=0)

    def median_blur(self, image, ksize=3):
        return cv2.medianBlur(image, ksize=ksize)

    def adaptive_thresh(self,
                        image,
                        maxValue=255,
                        method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        threshType=cv2.THRESH_BINARY_INV,
                        blocksize=13,
                        C=3):
        """
        Did not perform well, adjustments could possibly be made to reduce the noise in the images.
        """
        return cv2.adaptiveThreshold(
            image, maxValue, method, threshType, blocksize, C)
