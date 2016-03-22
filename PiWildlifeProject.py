import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.stats
import itertools
import datetime
import time
import os


class SingleSession():
    def __init__(self, average_span=5, observation_span=100, period=2,
                 p_threshold=0.95):
        self.observation_span = observation_span
        self.average_span = average_span
        self.data_store = "data"
        self.period = period
        self.p_threshold = p_threshold

        if os.path.isdir(self.data_store) is False:
            os.mkdir(self.data_store)

        self.capture_background()
        self.calculate_background()
        self.observation_mode()

    def capture_image(self):
        """ Capture an image and save it to disk, return the file name """
        compression = 0
        frame_skip = 2
        resolution = "1280x720"
        fn = "{}/{}.png".format(
            self.data_store,
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        os.system('fswebcam -q -r {} --greyscale --png {} {} -S {}'.format(
            resolution, compression, fn, frame_skip))
        print("Saved image {}".format(fn))
        return fn

    def get_image(self):
        """ Return np.array of image capture now """
        fn = self.capture_image()
        return fn, mpimg.imread(fn)

    def capture_background(self):
        print("Capturing average background image")
        start_time = time.time()
        images = []
        while time.time() - start_time < self.average_span:
            fn, img = self.get_image()
            images.append(img)
            self.remove_image(fn)
        number_of_images = len(images)

        background_img = np.zeros(images[0].shape)
        for img in images:
            background_img += img
        self.background_img = background_img / float(number_of_images)

        self.images = images

    def calculate_background(self):
        """ Calculate the statistics of the background residual"""

        self.background_mean = np.mean(
            [np.mean(np.abs(i - self.background_img)) for i in self.images])
        self.background_std = np.mean(
            [np.std(np.abs(i - self.background_img)) for i in self.images])

    def beta_cdf(self, x):
        return scipy.stats.beta.cdf(x, self.background_beta_fit[0],
                                    self.background_beta_fit[1])

    def classify_interest(self, img):
        res = np.abs(self.background_img - img)
        Rchannel = res[:, :, 0].flat
        sorted = np.sort(Rchannel)
        yvals = np.arange(len(sorted))/float(len(sorted))
        xloc = self.background_mean + 10 * self.background_std
        idx = np.argmin(np.abs(sorted - xloc))
        if yvals[idx] < self.p_threshold:
            print("Image classified as interesting with p={}".format(yvals[idx]))
            return True
        else:
            return False

    def observation_mode(self):
        print("Starting observation mode..")
        start_time = time.time()
        while time.time() - start_time < self.observation_span:
            fn, img = self.get_image()
            if self.classify_interest(img) is False:
                self.remove_image(fn)
            else:
                continue
            time.sleep(self.period)

    def remove_image(self, fn):
        os.remove(fn)


if __name__ == "__main__":
    SingleSession()
