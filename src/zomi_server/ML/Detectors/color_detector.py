"""
This will take an image and return the dominant color which obviously requires a color image.
Use ML to find a car, truck, etc. crop the bounding box and pass the image to this function to get the dominant color.
Understand that if CCTV IR is on, you will only get black/white/grey spectrum.
"""

import time
from logging import getLogger
from typing import List, Dict
from warnings import warn


try:
    import cv2
except ImportError:
    warn("OpenCV not installed!")
    raise
try:
    import cupy as cp  # Try importing CuPy
    _HAS_CUPY = True
    np = cp  # Dynamically alias CuPy as `np`
except ImportError:
    import numpy as np  # Fallback to NumPy
    _HAS_CUPY = False
from sklearn.cluster import KMeans
import webcolors

from ...Log import SERVER_LOGGER_NAME

LP: str = "color detect:"
logger = getLogger(SERVER_LOGGER_NAME)


class ColorDetector:
    kmeans: KMeans

    def __init__(self, config):
        top_n = config.top_n or 4
        _start = time.time()
        self.kmeans = KMeans(n_clusters=top_n, random_state=0, n_init=10)
        _end = time.time()
        self.config = config
        logger.info(f"perf:{LP} init of kmeans took {_end - _start:.5f} seconds")

        self.hex2names = webcolors.HTML4_HEX_TO_NAMES
        if config.spec == "css2":
            self.hex2names = webcolors.CSS2_HEX_TO_NAMES
        elif config.spec == "css21":
            self.hex2names = webcolors.CSS21_HEX_TO_NAMES
        elif config.spec == "css3":
            self.hex2names = webcolors.CSS3_HEX_TO_NAMES
        logger.debug(f"{LP} using webcolors spec: {config.spec}")

    async def warm_up(self):
        # create a random image to warm up Kmeans
        _start = time.time()
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        await self.detect(image)
        logger.debug(f"{LP} Warmup kmeans took {time.time() - _start:.5f} seconds")

    # 50x50 resize: Dominant colors (RGB): ['dimgray', 'lightsteelblue', 'black', 'lightgray', 'gray', 'darkslategray'] -- TOTAL: 4 sec
    # original size: Dominant colors (RGB): ['lightsteelblue', 'darkslategray', 'dimgray', 'gray', 'black', 'lightgray'] -- TOTAL: 15.1 sec
    async def detect(self, image: np.ndarray):
        # Resize the image to 50x50 if over 50x50
        if image.shape[0] > 50 and image.shape[1] > 50:
            image = cv2.resize(image, (50, 50))
        elif image.shape[0] > 50 or image.shape[1] > 50:
            if image.shape[0] > 50:
                image = cv2.resize(image, (50, image.shape[1]))
            else:
                image = cv2.resize(image, (image.shape[0], 50))

        # Reshape the image to a 2D array of pixels and 3 color values (RGB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        # Apply KMeans clustering to find the dominant color
        self.kmeans.fit(image)
        # Get the most dominant color (with the highest number of points in the cluster)
        unique, counts = np.unique(self.kmeans.labels_, return_counts=True)
        total = sum(counts)
        percentage = [(c / total) for c in counts]
        unique_counts = {a: b for a, b in zip(unique, percentage)}
        # logger.debug(f"{LP} {unique_counts=}")
        dominant_colors = self.kmeans.cluster_centers_[unique]
        dominant_colors = [tuple(map(int, color)) for color in dominant_colors]
        names = self.get_color_name(dominant_colors, percentage)
        # sort the dict by value ( { name: percent } )
        ret_ = dict(sorted(names.items(), key=lambda item: item[1], reverse=True))
        logger.debug(f"{LP} Dominant colors (sorted Name:%): {ret_}")
        return ret_

    def closest_color(self, requested_color: tuple):
        min_colors = {}
        for key, name in self.hex2names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[rd + gd + bd] = name
        return min_colors[min(min_colors.keys())]

    def get_color_name(
        self, requested_colors: List[tuple], percentages: List[float]
    ) -> Dict[str, float]:
        final = {}
        color_names = []
        color_percentages = []
        for i, req_color in enumerate(requested_colors):
            assert len(req_color) == 3, "Color must be a tuple of 3 elements"
            try:
                color_name = webcolors.rgb_to_name(req_color)
            except ValueError:
                color_name = self.closest_color(req_color)
            percentage = percentages[i]
            if color_name in color_names:
                existing_perc = color_percentages[color_names.index(color_name)]
                new_perc = percentage + existing_perc
                color_percentages[color_names.index(color_name)] = new_perc
            else:
                color_names.append(color_name)
                color_percentages.append(percentage)

        for name, perc in zip(color_names, color_percentages):
            final[name] = perc
        return final
