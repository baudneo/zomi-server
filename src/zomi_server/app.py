from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from platform import python_version
from typing import Union, List, Optional, Any, TYPE_CHECKING, Annotated, Tuple

import uvloop
from fastapi.exceptions import RequestValidationError
from pydantic import Base64Bytes, Field, BaseModel, Base64Str
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse


try:
    import cv2
except ImportError:
    cv2 = None
    raise ImportError(
        "OpenCV is not installed, please install it by compiling it for "
        "CUDA/cuDNN GPU support/OpenVINO Intel CPU/iGPU/dGPU support or "
        "quickly for only cpu support using 'opencv-contrib-python' package"
    )
import numpy as np

import pydantic
import uvicorn

from fastapi import (
    FastAPI,
    HTTPException,
    __version__ as fastapi_version,
    UploadFile,
    File,
    Body,
    Depends,
    Form,
)
from fastapi.responses import RedirectResponse, Response
from fastapi.security import OAuth2PasswordRequestForm

from .Models.Enums import ModelType, ModelFrameWork, ModelProcessor
from .Log import SERVER_LOGGER_NAME, SERVER_LOG_FORMAT
from .Log.handlers import BufferedLogHandler
from . import __version__
from .Models.config import (
    BaseModelConfig,
    GlobalConfig,
    Settings,
    APIDetector,
    LoggingSettings,
    DetectionResults,
    Result,
)
from .auth import *

logger = logging.getLogger(SERVER_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
buffered_log_handler = BufferedLogHandler()
buffered_log_handler.setFormatter(SERVER_LOG_FORMAT)
logger.addHandler(buffered_log_handler)
# Control uvicorn logging, what a mess!
uvi_logger = logging.getLogger("uvicorn")
uvi_error_logger = logging.getLogger("uvicorn.error")
uvi_access_logger = logging.getLogger("uvicorn.access")
uvi_loggers = (uvi_logger, uvi_error_logger, uvi_access_logger)
for _ul in uvi_loggers:
    _ul.setLevel(logging.DEBUG)
    _ul.propagate = False

logger.info(
    f"ZoMi MLAPI: {__version__} [Python: {python_version()} - "
    f"OpenCV: {cv2.__version__} - Numpy: {np.__version__} - FastAPI: {fastapi_version} - "
    f"Pydantic: {pydantic.VERSION}]"
)

app = FastAPI(
    debug=True,
    title="Machine Learning API - © dAIngerous consulting 2023",
    version=__version__,
    description="A blazing fast API for running Machine Learning models on images",
)

g: Optional[GlobalConfig] = None
LP: str = "mlapi:"
ANNOTATE_PATH: str = "/annotate"
DETECT_PATH: str = "/detect"
SLATE_COLORS: List[Tuple[int, int, int]] = [
    (39, 174, 96),
    (142, 68, 173),
    (0, 129, 254),
    (254, 60, 113),
    (243, 134, 48),
    (91, 177, 47),
]


def create_logs() -> logging.Logger:
    formatter = SERVER_LOG_FORMAT
    logger = logging.getLogger(SERVER_LOGGER_NAME)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    for _ul in uvi_loggers:
        _ul.setLevel(logging.DEBUG)
        _ul.addHandler(console_handler)

    return logger


def init_logs(config: Settings) -> None:
    """Initialize the logging system."""
    import getpass
    import grp
    import os

    lp: str = "init:logs:"
    sys_user: str = getpass.getuser()
    sys_gid: int = os.getgid()
    sys_group: str = grp.getgrgid(sys_gid).gr_name
    sys_uid: int = os.getuid()

    cfg: LoggingSettings = config.logging
    root_level = cfg.level
    logger.debug(
        f"{lp} Setting root logger level to {logging._levelToName[root_level]}"
    )
    logger.setLevel(root_level)
    for _ul in uvi_loggers:
        _ul.setLevel(root_level)

    if cfg.console.enabled is False:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                if h.stream == sys.stdout:
                    logger.info(f"{lp} Removing console log output!")
                    logger.removeHandler(h)

    if cfg.file.enabled:
        if cfg.file.file_name:
            _filename = cfg.file.file_name
        else:
            from .Models.DEFAULTS import DEF_SRV_LOG_FILE_FILENAME

            _filename = DEF_SRV_LOG_FILE_FILENAME
        abs_logfile = (cfg.file.path / _filename).expanduser().resolve()
        try:
            if not abs_logfile.exists():
                logger.info(f"{lp} Creating log file [{abs_logfile}]")
                from .Models.DEFAULTS import DEF_SRV_LOGGING_FILE_CREATE_MODE

                abs_logfile.touch(exist_ok=True, mode=DEF_SRV_LOGGING_FILE_CREATE_MODE)
            else:
                # Test if read/write permissions are available
                with abs_logfile.open(mode="a") as f:
                    pass
        except PermissionError:
            logger.warning(
                f"{lp} Logging to file disabled due to permissions"
                f" - No write access to '{abs_logfile.as_posix()}' for user: "
                f"{sys_uid} [{sys_user}] group: {sys_gid} [{sys_group}]"
            )
        else:
            # todo: add timed rotating log file handler if configured (for systems without logrotate)
            file_handler = logging.FileHandler(abs_logfile.as_posix(), mode="a")
            # file_handler = logging.handlers.TimedRotatingFileHandler(
            #     file_from_config, when="midnight", interval=1, backupCount=7
            # )
            file_handler.setFormatter(SERVER_LOG_FORMAT)
            if cfg.file.level:
                logger.debug(f"File logger level CONFIGURED AS {cfg.file.level}")
                # logger.debug(f"Setting file log level to '{logging._levelToName[g.config.logging.file.level]}'")
                file_handler.setLevel(cfg.file.level)
            logger.addHandler(file_handler)
            for _ul in uvi_loggers:
                _ul.addHandler(file_handler)

            # get the buffered handler and call flush with file_handler as a kwarg
            # this will flush the buffer to the file handler
            for h in logger.handlers:
                if isinstance(h, BufferedLogHandler):
                    logger.debug(f"Flushing buffered log handler to file")
                    h.flush(file_handler=file_handler)
                    # Close the buffered handler
                    h.close()
                    break
            logger.debug(
                f"Logging to file '{abs_logfile}' with user: "
                f"{sys_uid} [{sys_user}] group: {sys_gid} [{sys_group}]"
            )
    if cfg.syslog.enabled:
        # enable syslog logging
        syslog_handler = logging.handlers.SysLogHandler(
            address=cfg.syslog.address,
        )
        syslog_handler.setFormatter(SERVER_LOG_FORMAT)
        if cfg.syslog.level:
            logger.debug(
                f"Syslog logger level CONFIGURED AS {logging._levelToName[cfg.syslog.level]}"
            )
            syslog_handler.setLevel(cfg.syslog.level)
        logger.addHandler(syslog_handler)
        logger.debug(f"Logging to syslog at {cfg.syslog.address}")

    logger.info(f"Logging initialized...")


def get_global_config() -> GlobalConfig:
    return g


def create_global_config() -> GlobalConfig:
    """Create the global config object"""
    from .Models.config import GlobalConfig

    global g
    if not isinstance(g, GlobalConfig):
        g = GlobalConfig()
    return get_global_config()


def get_settings() -> Settings:
    return get_global_config().config


def get_available_models() -> List[BaseModelConfig]:
    return get_global_config().available_models


def locks_enabled():
    return get_settings().locks.enabled


def normalize_id(_id: str) -> str:
    return _id.strip().casefold()


def get_model(model_hint: Union[str, BaseModelConfig]) -> BaseModelConfig:
    """Get a model based on the hint provided. Hint can be a model name, model id, or a model object"""
    logger.debug(f"get_model: hint TYPE: {type(model_hint)} -> {model_hint}")
    available_models = get_available_models()
    if available_models:
        if isinstance(model_hint, BaseModelConfig):
            for model in available_models:
                if model.id == model_hint.id or model == model_hint:
                    return model
        elif isinstance(model_hint, str):
            for model in available_models:
                identifiers = {normalize_id(model.name), str(model.id)}
                logger.debug(f"get_model: identifiers: {identifiers}")
                if normalize_id(model_hint) in identifiers:
                    return model
    raise HTTPException(status_code=404, detail=f"Model {model_hint} not found")


async def detect(
    _model_hints: List[str],
    images: List[UploadFile],
    return_image: bool = False,
) -> List[List[Optional[DetectionResults]]]:
    available_models = get_available_models()
    _model_hints = [
        normalize_id(model_hint)
        for model_hint in _model_hints
        if model_hint is not None
    ]
    # logger.debug(f"threaded_detect: model_hints -> {_model_hints}")
    detectors: List[APIDetector] = []
    detections: List[Optional[List[DetectionResults]]] = []

    found_models = []
    for model in available_models:
        identifiers = {model.name, str(model.id)}
        if any(
            [normalize_id(model_hint) in identifiers for model_hint in _model_hints]
        ):
            detector = get_global_config().get_detector(model)
            detectors.append(detector)
            found_models.append(f"<'{model.name}' ({model.id})>")

    if found_models:
        logger.info(f"Found models: {found_models}")
        # check if images contains bytes or UploadFile objects
        if not isinstance(images[0], bytes):
            images = [
                load_image_into_numpy_array(await image.read()) for image in images
            ]
        else:
            images = [load_image_into_numpy_array(image) for image in images]
        for image in images:
            img_dets = []
            for detector in detectors:
                # logger.info(f"Starting detection for {detector}")
                img_dets.append(await detector.detect(image))
            detections.append(img_dets)
        ##### ThreadPool
        # futures = []
        # import concurrent.futures
        #
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     for image in images:
        #         for detector in detectors:
        #             # logger.info(f"Starting detection for {detector}")
        #             futures.append(executor.submit(detector.detect, image))
        # for future in futures:
        #     detections.append(future.result())
        # logger.info(f"{LP} ThreadPool detections -> {detections}")
    else:
        # return supplied images
        if not isinstance(images[0], bytes):
            images = [x.read() for x in images]
    if return_image:
        return detections, images
    return detections


def load_image_into_numpy_array(data: bytes) -> np.ndarray:
    """Load an uploaded image into a numpy array

    :param data: The image data in bytes
    :return: A cv2 imdecoded numpy array
    """
    np_img = np.frombuffer(data, np.uint8)
    if np_img is None:
        raise RuntimeError("Failed to create numpy array from image data")
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return frame


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")


@app.get("/models/available/all", summary="Get a list of all available models")
async def available_models_all():
    """FastAPI endpoint. Get a list of all available models"""
    try:
        logger.debug(f"About to try and grab available models....")
        x = {"models": get_global_config().available_models}
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        raise e
    else:
        logger.debug(f"Got available models: {x}")
        return x


@app.get(
    "/models/available/type/{model_type}",
    summary="Get a list of available models based on the type",
)
async def available_models_type(model_type: ModelType):
    """FastAPI endpoint. Get a list of available models based on the model type: obj, face, alpr"""
    available_models = get_global_config().available_models
    return {
        "models": [
            model.model_dump()
            for model in available_models
            if model.type_of == model_type
        ]
    }


@app.get(
    "/models/available/proc/{processor}",
    summary="Get a list of available models based on the processor",
)
async def available_models_proc(processor: ModelProcessor):
    """FastAPI endpoint. Get a list of available models based on the processor: cpu, gpu, tpu, none"""
    logger.info(f"available_models_proc: {processor}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.model_dump()
            for model in available_models
            if model.processor == processor
        ]
    }


@app.get(
    "/models/available/framework/{framework}",
    summary="Get a list of available models based on the framework",
)
async def available_models_framework(framework: ModelFrameWork):
    """FastAPI endpoint. Get a list of available models based on the framework: torch, ort, trt, opencv, etc"""
    logger.info(f"available_models_proc: {framework}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.model_dump()
            for model in available_models
            if model.framework == framework
        ]
    }


async def _proc_hints(
    hints: Union[None, str, List[Optional[str]]]
) -> List[Optional[str]]:
    splt_: Optional[List[str]] = None
    og: Union[str, None, List[Optional[str]]] = None

    if not hints:
        return []

    if isinstance(hints, str):
        # decode from json
        og = str(hints)
        if og.startswith("["):
            hints = json.loads(og)
    elif isinstance(hints, list):
        og = list(hints)
        # It's a list but is it JSON in the list?
        if len(og):
            for hint in og:
                if hint.startswith("["):
                    hints.extend(json.loads(hint))
                else:
                    hints.append(hint.strip("'\"[]").strip())

    hints = list(set(hints))
    if len(hints) == 1:
        # If there is only 1 hint, see if it can be split.
        splt_ = hints[0].split(",")
        if len(splt_) > 0:
            hints.extend(splt_)
    logger.debug(f"FINAL: hints: {hints } -- split: {splt_ = } -- orig: {og}")

    return hints


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    # or logger.error(f'{exc}')
    logger.error(f"{exc_str[:min(300, len(exc_str))] = }")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


class ImageRequest(BaseModel):
    # Using Base64Bytes will decode it for you back into bytes
    images: List[Optional[Base64Bytes]] = Field(
        default_factory=list, description="Images to run the ML model on"
    )
    hints_model: List[Optional[str]] = Field(None, alias="model_hints")


async def color_detect(detections, images):
    image_buffer = b""
    img_loop, i = 0, 0
    num_dets = len(detections)
    logger.debug(f"DBG>>> COLOR_DETECT: num_dets: {num_dets} ---- {detections = }")
    for img_detection, _image in zip(detections, images):
        img_loop += 1
        for detection in img_detection:
            i += 1
            logger.debug(f"DBG>>> COLOR_DETECT:detection: {detection} ({i}/{num_dets})")
            if detection and isinstance(detection, DetectionResults):
                det: Result
                for det in detection.results:
                    x1, y1, x2, y2 = det.bounding_box
                    # crop bounding box
                    crop = _image[y1:y2, x1:x2]
                    # detect color
                    colors = await get_global_config().color_detector.detect(crop)

            elif detection and not isinstance(detection, DetectionResults):
                raise RuntimeError(
                    f"DetectionResults object expected, got {type(detection)}"
                )
            else:
                logger.warning(f"DetectionResults object is {type(detection)}")

            is_success, img_buffer = cv2.imencode(".jpg", _image)
            if not is_success:
                raise RuntimeError("Failed to encode image")
            image_buffer = img_buffer.tobytes()

        # media_type here sets the media type of the actual response sent to the client.
        # Return the results in a header, this allows Swagger UI to display the image

        results_json = []
        for img_det in detections:
            for _det in img_det:
                results_json.append(_det.model_dump())

        return Response(
            content=image_buffer,
            media_type="image/jpeg",
            headers={"Results": json.dumps(results_json)},
            background=None,
        )


async def _annotate(detections, images) -> Response:
    image_buffer = b""
    img_loop, i = 0, 0
    num_dets = len(detections)
    logger.debug(f"DBG>>> ANNOTATE: num_dets: {num_dets} ---- {detections = }")
    num_dets = len(detections)
    for img_detection, _image in zip(detections, images):
        img_loop += 1
        for detection in img_detection:
            i += 1
            logger.debug(f"DBG>>> ANNOTATE:detection: {detection} ({i}/{num_dets})")
            if detection and isinstance(detection, DetectionResults):
                rand_color = SLATE_COLORS[random.randrange(len(SLATE_COLORS) - 1)]
                logger.debug(
                    f"DBG>>> ANNOTATE:detection: is a DetectionResults object, num results: {len(detection.results)}"
                )

                det: Result
                for det in detection.results:
                    x1, y1, x2, y2 = det.bounding_box
                    cv2.rectangle(_image, (x1, y1), (x2, y2), rand_color, 2)
                    cv2.putText(
                        _image,
                        f"{det.label} ({det.confidence:.2f})[{detection.name}]",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        rand_color,
                        2,
                    )

            elif detection and not isinstance(detection, DetectionResults):
                raise RuntimeError(
                    f"DetectionResults object expected, got {type(detection)}"
                )
            else:
                logger.warning(f"DetectionResults object is {type(detection)}")
            is_success, img_buffer = cv2.imencode(".jpg", _image)
            if not is_success:
                raise RuntimeError("Failed to encode image")
            image_buffer = img_buffer.tobytes()

        # media_type here sets the media type of the actual response sent to the client.
        # Return the results in a header, this allows Swagger UI to display the image

        results_json = []
        for img_det in detections:
            for _det in img_det:
                results_json.append(_det.model_dump())

        return Response(
            content=image_buffer,
            media_type="image/jpeg",
            headers={"Results": json.dumps(results_json)},
            background=None,
        )


@app.post(
    ANNOTATE_PATH,
    summary="Annotate Base64 encoded images using a set of models referenced by name/UUID",
    include_in_schema=False,
)
async def base64_annotate(
    image_request: ImageRequest = Body(
        ...,
        description="JSON body request with base64 encoded images and List of model names/UUIDs to run on the images",
    ),
):
    """FastAPI endpoint. Return an annotated jpeg image with bounding boxes and labels"""
    s = time.time()
    if not image_request:
        raise HTTPException(status_code=400, detail="No data provided (Empty Req)")
    hints = image_request.hints_model
    images = image_request.images
    images = [x for x in images if x is not None]
    if hints and images:
        detections = await detect(hints, images)
        if detections:
            logger.debug(f"{LP} {DETECT_PATH} results: {detections}")
        else:
            logger.debug(f"{LP} {DETECT_PATH} NO RESULTS!")
    elif not hints:
        raise HTTPException(status_code=400, detail="No model hints specified")
    elif not images:
        raise HTTPException(status_code=400, detail="No images specified")
    else:
        raise HTTPException(status_code=400, detail="No models or images specified")

    ret_ = await _annotate(detections, images)
    logger.debug(f"perf: {ANNOTATE_PATH} took {time.time() - s:.5f} s")

    return ret_


@app.post(
    DETECT_PATH,
    summary="Detect objects in Base64 encoded images using a set of models referenced by name/UUID",
    include_in_schema=False,
)
async def base64_detection(
    request: Request,
    images: Union[Optional[Base64Bytes], List[Optional[Base64Bytes]]] = Body(
        ...,
        description="JSON Form request with base64 encoded images and List of model names/UUIDs to run on the images",
    ),
    hints_model: Optional[str] = Form(
        None,
        description="List of model names/UUIDs to run on the images",
        alias="model_hints",
    ),
):
    """FastAPI endpoint. Detect objects in images using a set of threaded models referenced by name/UUID. Images are sent as Form data and base64 encoded"""
    start = time.time()
    # if not image_request:
    #     raise HTTPException(status_code=400, detail="No data provided (Empty Req)")
    # hints = image_request.hints_model
    # images = image_request.images
    if isinstance(images, list):
        logger.debug(f"Got a list of images")
        images = [x for x in images if x is not None]
    elif isinstance(images, bytes):
        logger.debug(f"Got a single image as bytes, converting to a list")
        images = [images]
    hints = hints_model
    hints = json.loads(hints)
    logger.debug(f"Got {hints_model = }")
    if hints and images:
        detections = await detect(hints, images)
        if detections:
            logger.debug(f"{LP} {DETECT_PATH} results: {detections}")
        else:
            logger.debug(f"{LP} {DETECT_PATH} NO RESULTS!")
    elif not hints:
        raise HTTPException(status_code=400, detail="No model hints specified")
    elif not images:
        raise HTTPException(status_code=400, detail="No images specified")
    else:
        raise HTTPException(status_code=400, detail="No models or images specified")

    logger.info(f"perf:{LP} '{DETECT_PATH}' path took {time.time() - start:.5f} s")

    return detections


class MLAPI:
    cached_settings: Settings
    cfg_file: Path
    server: uvicorn.Server
    color_detector: Any

    def __init__(self, cfg_file: Union[str, Path], run_server: bool = False):
        """
        Initialize the FastAPI MLAPI server object, read a supplied YAML config file, and start the server if requested.
        :param cfg_file: The config file to read.
        :param run_server: Start the server after initialization.
        """
        create_global_config()
        if not isinstance(cfg_file, (str, Path)):
            raise TypeError(
                f"The YAML config file must be a str or pathlib.Path object, not {type(cfg_file)}"
            )
        # test that the file exists and is a file
        self.cfg_file = Path(cfg_file)
        if not self.cfg_file.exists():
            raise FileNotFoundError(f"'{self.cfg_file.as_posix()}' does not exist")
        elif not self.cfg_file.is_file():
            raise TypeError(f"'{self.cfg_file.as_posix()}' is not a file")
        self.read_settings()
        if run_server:
            self.start()

    def read_settings(self) -> Settings:
        """Read the YAML config file,  initialize the user_db, set the global config (g.config) object and load models"""
        logger.info(f"reading settings from '{self.cfg_file.as_posix()}'")
        from .Models.config import parse_client_config_file

        self.cached_settings = parse_client_config_file(self.cfg_file)
        from .auth import UserDB

        get_global_config().user_db = UserDB(self.cached_settings.server.auth.db_file)
        get_global_config().config = self.cached_settings
        init_logs(self.cached_settings)
        # create async locks for CPU, GPU, TPU
        locks_config = self.cached_settings.locks
        get_global_config().async_locks = {
            ModelProcessor.CPU: asyncio.BoundedSemaphore(locks_config.cpu.max),
            ModelProcessor.GPU: asyncio.BoundedSemaphore(locks_config.gpu.max),
            ModelProcessor.TPU: asyncio.BoundedSemaphore(locks_config.tpu.max),
        }
        loop = uvloop.new_event_loop()
        if (
            get_global_config().config.color
            and get_global_config().config.color.enabled
        ):
            from .ML.Detectors.color_detector import ColorDetector

            logger.info(f"Creating global color detector...")
            get_global_config().color_detector = ColorDetector(
                get_global_config().config.color
            )
            logger.info(f"Starting to warmup color detector in uvloop task...")
            task = loop.create_task(get_global_config().color_detector.warm_up())
            # execute task
            loop.run_until_complete(task)

        available_models = get_global_config().available_models = (
            self.cached_settings.available_models
        )
        if available_models:
            logger.info(f"Starting to load models...")
            timer = time.time()

            _method = "Async+Executor"
            with ThreadPoolExecutor() as executor:
                for model in available_models:
                    # loop = uvloop.new_event_loop()
                    loop.run_in_executor(
                        executor, get_global_config().create_detector, model
                    )
            logger.info(
                f"perf:{LP} '{_method}' method of loading models took {time.time() - timer:.5f} s"
            )
            warmup_start = time.time()
            # warm up synchronously, to avoid OOM issues on startup.
            # todo: make model loading configurable to be async or sync. async is much faster but,
            #  if you don't have much RAM or VRAM, OOM issues can happen.
            for model in available_models:
                __detector = get_global_config().get_detector(model)
                __detector.warm_up()
        else:
            logger.warning(
                f"No models found in config file! skipping loading of models..."
            )

        return self.cached_settings

    def restart(self):
        self.server.shutdown()
        self.read_settings()
        self.start()

    @staticmethod
    def _get_cf_ip_list() -> List[str]:
        """Grab cloudflare ip ranges for IPv4/v6 and store them in a List of strings"""
        import requests

        cf_ipv4_url = "https://www.cloudflare.com/ips-v4"
        cf_ipv6_url = "https://www.cloudflare.com/ips-v6"
        cf_ipv4 = requests.get(cf_ipv4_url).text.splitlines()
        cf_ipv6 = requests.get(cf_ipv6_url).text.splitlines()
        cf_ips = cf_ipv4 + cf_ipv6
        return cf_ips

    @staticmethod
    def _breakdown_subnets(ip_list: List[str]) -> List[str]:
        """Breakdown a list of IP ranges into individual IPs

        gunicorn < 22.0 does not support subnets in forwarded_allow_ips"""
        ips = []
        if ip_list:
            logger.debug(f"Breaking down subnets: {ip_list}")

            # Using list comprehension to process multiple IP ranges concurrently
            ips = [
                str(ip)
                for ip_range in ip_list
                if "/" in ip_range
                for ip in ipaddress.ip_network(ip_range).hosts()
            ]

            logger.debug(f"Subnet breakdown: {ips}")
        return ips

    def start(self):
        _avail = {}
        forwarded_allow = []

        _proxy_headers = get_global_config().config.uvicorn.proxy_headers
        if _proxy_headers:
            forward_ips = self.cached_settings.uvicorn.forwarded_allow_ips
            if forward_ips:
                forwarded_allow = [str(x) for x in forward_ips if x]
                logger.debug(f"Added {len(forwarded_allow)} forwarded_allow hosts")

            # if self.cached_settings.uvicorn.grab_cloudflare_ips:
            #     forwarded_allow += self._get_cf_ip_list()
            #     logger.debug(
            #         f"Grabbed Cloudflare IP ranges to append to forwarded_allow hosts"
            #     )
            #     logger.debug(f"\n\nforwarded_allow: {forwarded_allow}\n\n{_proxy_headers = }\n")
        avail_models = get_global_config().available_models
        if avail_models:
            for model in get_global_config().available_models:
                _avail[normalize_id(model.name)] = str(model.id)
            logger.info(f"AVAILABLE MODELS! --> {_avail}")
        server_cfg = get_global_config().config.server
        # This takes forever when breaking down cloudflare ip subnets.
        # Wait for
        # forwarded_allow = self._breakdown_subnets(forwarded_allow)
        config = uvicorn.Config(
            app="zomi_server.app:app",
            host=str(server_cfg.address),
            port=server_cfg.port,
            forwarded_allow_ips=forwarded_allow,
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
            },
            log_level="debug",
            proxy_headers=_proxy_headers,
            reload=self.cached_settings.uvicorn.reload,
            reload_dirs=self.cached_settings.uvicorn.reload_dirs,
        )
        self.server = uvicorn.Server(config=config)
        try:
            import uvloop

            loop = uvloop.new_event_loop()
            self.server.run()
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt, shutting down")
        except BrokenPipeError:
            logger.info("Broken Pipe, shutting down")
        except Exception as e:
            logger.exception(f"Shutting down because of Exception: {e}")
        finally:
            logger.info("Shutting down cleanly in finally: logic")
            loop.run_until_complete(self.server.shutdown())


# AUTH STUFF
@app.post(
    "/login",
    response_model=Token,
    summary="Login to get an authentication token",
)
async def login_for_access_token(
    request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    """FastAPI endpoint. Login to get an authentication token"""

    ip = request.client.host
    if "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
    if "x-real-ip" in request.headers:
        ip = request.headers["x-real-ip"]
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    logger.debug(f"{LP} Access token requested by: {form_data.username} using IP: {ip}")
    user = get_global_config().user_db.authenticate_user(
        form_data.username, form_data.password, ip=ip
    )

    if not user:
        raise credentials_exception
    access_token = create_access_token(data={"sub": user.username, "roles": user.roles})

    return access_token


@app.post(
    f"/swagger{ANNOTATE_PATH}",
    # Set what the media type will be in the autogenerated OpenAPI specification.
    # https://fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
    responses={200: {"content": {"image/jpeg": {}}}},
    # Prevent FastAPI from adding "application/json" as an additional
    # response media type in the autogenerated OpenAPI specification.
    # https://github.com/tiangolo/fastapi/issues/3258
    response_class=Response,
    summary="Return an annotated jpeg image with bounding boxes and labels",
)
async def swagger_annotate(
    user_obj=Depends(verify_token),
    hints_: List[str] = Body(
        ...,
        openapi_examples=["yolov4", "yolov4 tiny", "yolov7-tiny"],
        description="A list of model names/UUIDs",
    ),
    image: UploadFile = File(...),
):
    """FastAPI endpoint. Return an annotated jpeg image with bounding boxes and labels"""
    s = time.time()
    hints = await _proc_hints(hints_)
    if hints:
        detections, images = await detect(hints_, [image], return_image=True)
        ret_ = await _annotate(detections, images)
        logger.debug(f"perf: {ANNOTATE_PATH} took {time.time() - s:.5f} s")

        return ret_


@app.post(
    f"/swagger{DETECT_PATH}",
    summary="Detect objects in images using a set of models referenced by name/UUID",
)
async def swagger_detection(
    request: Request,
    user_obj=Depends(verify_token),
    hints_model: List[Optional[str]] = Body(
        ...,
        description="List of model names/UUIDs",
        examples=["yolo-nas-l", "97acd7d4-270c-4667-9d56-910e1510e8e8", "yolov7 tiny"],
        alias="model_hints",
    ),
    images: List[UploadFile] = File(..., description="Images to run the ML model on"),
):
    """FastAPI endpoint. Detect objects in images using a set of threaded models referenced by name/UUID"""
    start = time.time()
    if not images:
        raise HTTPException(status_code=400, detail="No image(s) provided")
    hints = await _proc_hints(hints_model)
    if hints:
        detections = await detect(hints, images)
        if detections:
            logger.debug(f"{LP} {DETECT_PATH} results: {detections}")
        else:
            logger.debug(f"{LP} {DETECT_PATH} NO RESULTS!")
    else:
        detections = {"error": "No models specified"}
    logger.info(f"perf:{LP} {DETECT_PATH} took {time.time() - start:.5f} s")
    return detections
