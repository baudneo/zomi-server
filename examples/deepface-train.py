#! /usr/bin/env python3
"""
A script to extract facial embeddings from images to store on disk for facial recognition
via the deepface python library. This will train each person(s) image(s) on each recognition model and
store the embeddings. If there is more than one image for a person, the embeddings will be
averaged to create a single embedding for that person.

It is required to have a directory structure like:
facial_data (root directory)
├── Alice (name of the person)
│   ├── Alice.jpg (name of the images does not matter to the script)
├── Bob
│   ├── Bob1.jpg
│   ├── Bob2.jpg

A dictionary is used to store the embeddings of each person and each supported model.

all_vectors = {
    "Alice": {
        "VGG-Face": {
            "all": [vector1],
            "average": [vector]
        },
        "Facenet": {
            "all": [vector1],
            "average": [vector]
    },
    "Bob": {
        "VGG-Face": {
            "all": [vector1, vector2],
            "average": [vector],
        },
        "Facenet": {
            "all": [vector1, vector2],
            "average": [vector],
        },
    },
    ...
}

We will use the deepface library to extract the facial embeddings from the images. If more than one
image is provided for a person, we will average the embeddings to create a single embedding for that person.

"""

import argparse
import math
import os
import pickle
import sys
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons.folder_utils import get_deepface_home, initialize_folder
from deepface.models.FacialRecognition import FacialRecognition

log_formatter = logging.Formatter(
    "%(asctime)s.%(msecs)d %(name)s[%(process)s] %(levelname)s line:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
logger = logging.getLogger("df-train")
console = logging.StreamHandler(stream=sys.stdout)
console.setFormatter(log_formatter)
logger.addHandler(console)
MODELS = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
]
DETECTORS = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yunet",
    "centerface",
]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Use deepface to extract facial embeddings from images for face recognition. "
                    "This will iterate through all persons, images and deepface models."
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing the facial data",
        type=str,
    )
    parser.add_argument(
        "--model-dir",
        "-M",
        help="Directory to store deepface models "
        "(same place as zomi-server models!)",
        required=True,
        dest="model_dir",
        type=str,
    )
    parser.add_argument(
        "--detector-backend",
        "-B",
        help="Face detector backend (opencv / ssd are fast, mtcnn / retinaface are accurate)",
        default="opencv",
        choices=DETECTORS,
    )
    parser.add_argument(
        "--override",
        "-o",
        help="Override existing embeddings (if any)",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        "-d",
        "-D",
        help="Enable debug logging",
        action="store_true",
    )
    parser.add_argument(
        "--models",
        "-m",
        help="Deepface models to use",
        nargs="+",
        default=MODELS,
        choices=MODELS,

    )

    return vars(parser.parse_args())


def check_dir_for_embeddings(dir: Path) -> Optional[dict]:
    bgn = "./deepface_embeddings-*"
    existing_files = []
    for file in dir.glob(bgn):
        existing_files.append(file)
    if existing_files:
        existing_files.sort(key=lambda x: x.name.split("-")[-1], reverse=True)
        logger.info(
            f"Found existing embeddings: {existing_files}, loading the most recent"
        )
    return pickle.load(open(existing_files[0], "rb")) if existing_files else None


def checks():
    if not image_dir.exists():
        logger.error(f"Image directory does not exist: {image_dir}")
        sys.exit(1)
    elif not image_dir.is_dir():
        logger.error(f"Image directory is not a directory: {image_dir}")
        sys.exit(1)
    logger.debug(f"Image directory: {image_dir}")


if __name__ == "__main__":
    args: dict = parse_cli()
    for handler in logger.handlers:
        if args["debug"]:
            logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
        handler.setFormatter(log_formatter)
    logger.debug(f"CLI args: {args}")
    image_dir: Path = Path(args["image_dir"])
    model_dir: Path = Path(args["model_dir"])
    checks()
    detect_backend: str = args["detector_backend"]
    override: bool = args["override"]
    recog_models = args["models"]
    if not recog_models:
        logger.warning(f"No models specified, using all available models: {', '.join(MODELS).rstrip(',')}")
        recog_models = MODELS
    else:
        logger.info(f"Using configured models: {', '.join(recog_models).rstrip(',')}")
    if not detect_backend:
        logger.warning(f"No detector backend specified, using opencv")
        detect_backend = "opencv"
    else:
        logger.info(f"Using configured detector backend: {detect_backend}")
    output_file: Path = Path(
        f"./deepface_embeddings-{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    )
    logger.debug(f"Output file name calculated as: {output_file}")
    df_home_b4: str = get_deepface_home()
    os.environ["DEEPFACE_HOME"] = model_dir.expanduser().resolve().as_posix()
    logger.info(
        f"Overriding deepface model storage root directory > {df_home_b4} -> "
        f"{get_deepface_home()} (root should be same as zomi-server models!)"
    )
    initialize_folder()

    existing_vectors: Optional[dict] = check_dir_for_embeddings(
        Path("./").expanduser().resolve()
    )
    all_vectors: dict = {}
    for person in image_dir.iterdir():
        if person.is_dir():
            all_vectors[person.name] = {}
            for model in MODELS:
                if existing_vectors:
                    if person.name in existing_vectors:
                        if model in existing_vectors[person.name] and not override:
                            logger.warning(
                                f"Skipping model: {model} for person: {person.name} as embeddings already "
                                f"exist and the --override flag was not set"
                            )
                            all_vectors[person.name][model] = existing_vectors[
                                person.name
                            ][model]
                            continue
                        else:
                            logger.info(
                                f"Overriding existing model: {model} embeddings for person: {person.name}"
                            )
                    else:
                        logger.info(
                            f"Creating new person: {person.name} to store model embeddings"
                        )
                else:
                    logger.info(
                        f"No existing embeddings found, creating new person: {person.name}"
                    )
                vectors = []
                try:
                    model_repr: FacialRecognition = DeepFace.build_model(model)
                except Exception as e:
                    logger.error(
                        f"Error building model: {model} for person: {person.name} -> {e}"
                    )
                    continue
                model_shape = model_repr.output_shape
                for image in person.iterdir():
                    img = cv2.imread(image.expanduser().resolve().as_posix())
                    try:
                        results = DeepFace.represent(
                            img, model_name=model, detector_backend=detect_backend
                        )
                    except Exception as e:
                        logger.exception(
                            f"Error extracting facial embeddings for person: {person.name} "
                            f"using model: {model} in image: {image.name} -> {e}"
                        )
                        continue
                    else:

                        if results:
                            """
                            Returns: results (List[Dict[str, Any]]):
                            A list of dictionaries, each containing the following fields:
                              - embedding (List[float]): Multidimensional vector representing facial features. The number of dimensions varies based on the reference model (e. g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).
                              - facial_area (dict): Detected facial area by face detection in dictionary format. Contains 'x' and 'y' as the left-corner point, and 'w' and 'h' as the width and height. If `detector_backend` is set to 'skip', it represents the full image area and is nonsensical.
                              - face_confidence (float): Confidence score of face detection. If `detector_backend` is set to 'skip', the confidence will be 0 and is nonsensical
                            """
                            num_faces = len(results)
                            logger.info(
                                f"Extracted {num_faces} facial embedding(s) for person: {person.name} "
                                f"in image: {image.name} using model: {model} (vectors: {model_shape})"
                            )
                            for result in results:
                                embedding: List[float] = result["embedding"]
                                # { 'x': 0, 'y': 0, 'w': 0, 'h': 0 }
                                facial_area: dict = (
                                    result["facial_area"]
                                    if detect_backend != "skip"
                                    else None
                                )
                                face_confidence: float = (
                                    result["face_confidence"]
                                    if detect_backend != "skip"
                                    else None
                                )
                                vectors.append(result["embedding"])
                        else:
                            logger.warning(
                                f"No facial embeddings found for person: {person.name} using model: "
                                f"{model} in image: {image.name}"
                            )
                all_vectors[person.name][model] = {
                    "all": vectors,
                    "average": (
                        np.mean(np.array(vectors), axis=0).tolist()
                        if len(vectors) > 1
                        else vectors
                    ),
                }

    # Save the facial embeddings to a file
    try:
        with open(output_file, "wb") as f:
            pickle.dump(all_vectors, f)
    except Exception as e:
        logger.exception(
            f"Error saving facial embeddings to file: {output_file} -> {e}"
        )
    else:
        logger.info(f"Facial embeddings saved to {output_file}")

    logger.info("End of deepface training script")
