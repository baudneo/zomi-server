"""Add deepface facial data framework support"""

import os
from logging import getLogger
from typing import TYPE_CHECKING, Union, List

import numpy as np
import deepface.commons.folder_utils
from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition as DFModel
from deepface.modules.verification import find_euclidean_distance, find_threshold
from annoy import AnnoyIndex

from ...app import get_global_config
from ...Models.Enums import ModelProcessor
from ...Models.config import DetectionResults, Result
from ...Log import SERVER_LOGGER_NAME
from .ocr import OCRBase

if TYPE_CHECKING:
    from ...Models.config import (
        DeepFaceModelConfig,
        DeepFaceModelOptions,
        DeepFaceRecognitionOptions,
    )

logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "deepface:"


class DeepFaceWrapper:
    def __init__(self, config: "DeepFaceModelConfig"):
        # todo: hook into deepface logger

        self.config: DeepFaceModelConfig = config
        self.detect_options: DeepFaceModelOptions = config.detection_options
        self.recog_options: DeepFaceRecognitionOptions = config.recognition_options
        self.name = config.name
        self.processor: ModelProcessor = config.processor
        self._override_df_home()
        self.recog_model: DFModel = DeepFace.build_model(self.recog_options.input)

    def _override_df_home(self):
        """Try and override the DEEPFACE_HOME environment variable to store models in the configured models directory"""
        df_b4 = deepface.commons.folder_utils.get_deepface_home()
        os.environ["DEEPFACE_HOME"] = (
            get_global_config()
            .config.system.models_dir.expanduser()
            .resolve()
            .as_posix()
        )
        logger.debug(
            f"{LP} Attempting to override deepface model storage root directory. Before: {df_b4} - After: "
            f"{deepface.commons.folder_utils.get_deepface_home()}"
        )

    def _get_proc_str(self) -> str:
        ret_ = "gpu:{idx}".format(idx=self.config.gpu_idx)
        if self.processor == ModelProcessor.CPU:
            ret_ = "cpu:{idx}".format(idx=self.config.cpu_idx)
        return ret_

    @staticmethod
    def average_embeddings(vectors: List[np.ndarray]) -> np.ndarray:
        """Average a list of embeddings, used when more than 1 photo of a person is provided"""
        average_embedding = np.mean(vectors, axis=0)
        return average_embedding

    def detect(self, input_image: np.ndarray) -> DetectionResults:
        """
        Use deepface to detect if any faces are present in an image

        Args:
            input_image (np.ndarray): The image to detect faces in (BGR format)

        """
        result = None
        try:
            detected_faces = DeepFace.extract_faces(
                input_image,
                detector_backend=self.detect_options.backend,
                enforce_detection=False,
                align=self.detect_options.align,
                grayscale=self.detect_options.greyscale,
                anti_spoofing=self.detect_options.anti_spoofing,
                expand_percentage=0,
            )
        except Exception as e:
            logger.error(f"{LP} Error detecting faces: {e}")
            result.error = str(e)

    def recognize(self, *args, **kwargs) -> DetectionResults:
        return self.verify(*args, **kwargs)

    def compare_embeddings(
        self, embedding_1: Union[np.ndarray, List], embedding_2: Union[np.ndarray, List]
    ) -> bool:
        threshold = 0.6
        identified = False
        distance = find_euclidean_distance(embedding_1, embedding_2)
        auto_threshold = False
        if not self.recog_options.auto_threshold:
            if not self.recog_options.distance_threshold:
                logger.error(
                    f"{LP} No distance threshold set for model {self.recog_model.model_name}, using auto threshold"
                )
                auto_threshold = True
            else:
                threshold = self.recog_options.distance_threshold
        else:
            auto_threshold = True

        if auto_threshold:
            threshold = find_threshold(
                self.recog_model.model_name, self.recog_options.distance_metric
            )
            logger.debug(f"{LP} Auto threshold: {threshold}")

        if distance <= threshold:
            identified = True
        return identified

    def extract_embeddings(self, input_image: np.ndarray) -> Union[np.ndarray, None]:
        """Extract facial embeddings from an image using a facial recognition CNN model"""
        try:
            embeddings = DeepFace.represent(
                input_image,
                model_name=self.recog_model.model_name,
                detector_backend=self.detect_options.backend,
                align=self.detect_options.align,
            )
        except Exception as e:
            logger.error(f"{LP} Error extracting embeddings: {e}")
            embeddings = None
        logger.debug(f"{LP} Extracted embeddings: {embeddings}")
        return embeddings

    def ann_vector_search(
        self, input_embedding: np.ndarray, vectors: List[np.ndarray]
    ) -> Union[List[int], None]:
        """perform approximate nearest neighbors vector search on saved embeddings using annoy"""
        dim = len(vectors)
        index = AnnoyIndex(dim, "euclidean")
        for i in range(len(vectors)):
            index.add_item(i, vectors[i])
        num_trees = 10
        num_jobs = get_global_config().config.system.thread_workers or -1
        index.build(n_trees=num_trees, n_jobs=num_jobs)
        k = 5  # Number of nearest neighbors to search for
        indices, distances = index.get_nns_by_vector(
            input_embedding, k, include_distances=True
        )
        combined_data = list(zip(indices, distances))
        logger.debug(f"{LP} Nearest neighbors: {combined_data}")
        return indices
