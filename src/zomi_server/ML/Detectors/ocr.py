from __future__ import annotations

import asyncio
import functools
import time
from logging import getLogger
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any

import cv2
try:
    import cupy as cp  # Try importing CuPy
    _HAS_CUPY = True
    np = cp  # Dynamically alias CuPy as `np`
except ImportError:
    import numpy as np  # Fallback to NumPy
    _HAS_CUPY = False

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

from ...Log import SERVER_LOGGER_NAME

logger = getLogger(SERVER_LOGGER_NAME)

try:
    import pytesseract
except ImportError:
    pytesseract = None
    logger.warning(f"PyTesseract not installed!")
else:
    # check for existence of tesseract executable
    import shutil

    x = shutil.which("tesseract")
    if x is None:
        pytesseract = None
        logger.warning(
            f"PyTesseract is installed but it seems tesseract executable is not installed!"
        )


try:
    import easyocr
except ImportError:
    easyocr = None
    logger.warning(f"EasyOCR not installed!")

from ...Models.config import OCRConfig

LP: str = "OCR:"
loop: Optional[asyncio.AbstractEventLoop] = None
PaddleOCR: Optional[Any] = None


class OCRBase:
    ocr_cfg: Union[OCRConfig.EasyOCRConfig, OCRConfig.PaddleOCRConfig]
    ocr_readers: List[Optional[Union[easyocr.Reader, Any]]] = []
    ocr_cfg: OCRConfig
    _num: int = 1

    def __init__(self, config: OCRConfig):
        self.ocr_cfg = config
        self.LP = f"{LP}"
        if not config:
            logger.warning(f"{LP} No OCR config supplied, OCR will not be available")
        else:
            logger.debug(f"{LP} OCR config supplied, OCR will be available")
            asyncio.run(self.init_readers(config))

    async def init_readers(self, config: OCRConfig):
        global loop
        from ...app import get_global_config

        if loop is None or loop.is_closed():
            loop = asyncio.get_running_loop()

        _s = time.time()

        if config.easy_ocr.enabled is True:
            cfg = config.easy_ocr
            logger.debug(f"{LP} Instantiating EasyOCR (loading models)")
            kwargs = {
                "lang_list": cfg.lang,
                "gpu": cfg.gpu,
                "download_enabled": True,
                "detector": True,
                "recognizer": True,
                "recog_network": "standard",
                "model_storage_directory": (
                    get_global_config().config.system.variable_data_path
                    / ".ocr/easyocr/models"
                )
                .expanduser()
                .resolve()
                .as_posix(),
                "user_network_directory": (
                    get_global_config().config.system.variable_data_path
                    / ".ocr/easyocr/user_networks"
                )
                .expanduser()
                .resolve()
                .as_posix(),
            }
            s = time.time()
            self.ocr_readers.append(
                await loop.run_in_executor(
                    None, functools.partial(easyocr.Reader, **kwargs)
                )
            )

            logger.debug(
                f"{self.LP} EasyOCR Reader instantiated in {time.time() - s:.5f} seconds"
            )

        else:
            logger.debug(f"{LP} EasyOCR is disabled for '{self.name}'")

        if config.paddle_ocr.enabled is True:
            global PaddleOCR
            try:
                from paddleocr import PaddleOCR

                # from paddleocr.tools.infer.utility import draw_ocr
            except ImportError:
                PaddleOCR = None
                logger.warning(f"PaddleOCR not installed!")
            else:
                s = time.time()
                logger.debug(f"{LP} Instantiating PaddleOCR (loading models)")
                # need to run only once to download and load model into memory
                kwargs = {
                    "use_angle_cls": True,
                    "lang": "en",
                    # PaddleOCR will; raise an red herring exception if you place all models in same dir
                    # it is required to place det, rec and cls models in different dirs
                    "det_model_dir": (
                        get_global_config().config.system.variable_data_path
                        / ".ocr/paddle/det_models"
                    )
                    .expanduser()
                    .resolve()
                    .as_posix(),
                    "rec_model_dir": (
                        get_global_config().config.system.variable_data_path
                        / ".ocr/paddle/rec_models"
                    )
                    .expanduser()
                    .resolve()
                    .as_posix(),
                    "cls_model_dir": (
                        get_global_config().config.system.variable_data_path
                        / ".ocr/paddle/cls_models"
                    )
                    .expanduser()
                    .resolve()
                    .as_posix(),
                }
                self.ocr_readers.append(
                    await loop.run_in_executor(
                        None, functools.partial(PaddleOCR, **kwargs)
                    )
                )
                logger.debug(
                    f"{self.LP} PaddleOCR reader instantiated in {time.time() - s:.5f} seconds"
                )
        else:
            logger.debug(f"{LP} PaddleOCR is disabled for '{self.name}'")

        if config.tesseract.enabled is True:
            _ex: Optional[str] = None
            logger.debug(f"{LP} Checking for 'pytesseract' AND 'tesseract executable'")
            if pytesseract is None:
                logger.warning(
                    f"{LP} PyTesseract not installed, cannot use Tesseract OCR"
                )
            else:
                _ex = config.tesseract.executable
                # check for existence of tesseract executable
                if not _ex:
                    _ex = "tesseract"
                if isinstance(_ex, Path):
                    if not _ex.exists():
                        logger.warning(
                            f"{LP} PyTesseract is installed but it seems tesseract executable ('{_ex}') does not exist!"
                        )
                        _ex = "tesseract"
                else:
                    _ex = _ex.expanduser().resolve().as_posix()
                exc_inst = shutil.which(_ex)
                if not exc_inst:
                    logger.warning(
                        f"{LP} PyTesseract is installed but it seems tesseract executable ('{_ex}') is not installed!"
                    )
                else:
                    if _ex != "tesseract":
                        logger.debug(
                            f"{LP} Using custom tesseract executable: {exc_inst}"
                        )
                        pytesseract.pytesseract.tesseract_cmd = exc_inst
                    self.ocr_readers.append(pytesseract)
        else:
            logger.debug(f"{LP} Tesseract OCR is disabled for '{self.name}'")

        logger.debug(f"{LP} OCR readers initialized in {time.time() - _s:.5f} seconds")

    def auto_adjust_brightness_contrast(
        self,
        image,
        target_mean_intensity: Optional[int] = None,
        target_contrast: Optional[int] = None,
    ):
        """Auto adjust the brightness and contrast of an image.

        This function calculates the average pixel intensity of an image and
        adjusts the brightness and contrast to match a target intensity.

        Parameters
        ----------
        image : numpy.ndarray
            The image to adjust.
        target_mean_intensity : int
            The target mean intensity. The default is 127.
        target_contrast : int
            The target contrast. The default is 50.

        Returns
        -------
        numpy.ndarray
            The adjusted image.
        """
        # Calculate the average pixel intensity of the image
        average_intensity = np.mean(image)

        # Calculate the contrast of the image using standard deviation
        contrast = np.std(image)

        if target_mean_intensity is None:
            target_mean_intensity = 127
        if target_contrast is None:
            target_contrast = 50
        # Calculate the difference between the target and current mean intensity
        brightness_difference = target_mean_intensity - average_intensity

        # Calculate the contrast adjustment factor
        contrast_factor = target_contrast / contrast if contrast != 0 else 1

        # Apply brightness and contrast adjustments
        logger.debug(
            f"auto_adjust_brightness_contrast: {brightness_difference = } -- {contrast_factor = }"
        )
        adjusted_image = cv2.convertScaleAbs(
            image, alpha=contrast_factor, beta=brightness_difference
        )

        return adjusted_image

    def is_dark_image(self, image, threshold=100):
        # Calculate average pixel intensity
        average_intensity = np.mean(image)

        logger.debug(f"is_image_dark: Average intensity of image: {average_intensity}")

        return average_intensity < threshold

    async def stretch_histogram(self, image):
        # Apply histogram stretching to enhance contrast
        min_intensity = np.min(image)
        max_intensity = np.max(image)
        stretched_image = (
            255 * (image - min_intensity) / (max_intensity - min_intensity)
        )
        stretched_image = np.clip(
            stretched_image, 0, 255
        )  # Clip values to the valid range [0, 255]
        return stretched_image.astype(np.uint8)

    async def ocr(
        self, inp_image: np.ndarray, num: Optional[int] = None
    ) -> Optional[List[Tuple]]:
        global loop

        if loop is None:
            loop = asyncio.get_running_loop()
        if loop.is_closed():
            loop = asyncio.get_running_loop()

        if num is None:
            num = 1
        self._num = num
        ret_ = {}
        if self.ocr_cfg.debug_images:
            cv2.imwrite(
                (self.ocr_cfg.dbg_img_dir / f"0{num}_01-Orig.jpg").as_posix(),
                inp_image,
            )

        s = time.time()
        if self.ocr_readers:
            # resize before processing for better results
            resize_factor = 3
            # set resize factors based on image dimensions, smaller images need a bigger factor
            if inp_image.shape[0] < 200:
                resize_factor = 5
            elif inp_image.shape[0] < 400:
                resize_factor = 4
            elif inp_image.shape[0] < 800:
                resize_factor = 3
            elif inp_image.shape[0] < 1200:
                resize_factor = 2
            else:
                resize_factor = 1

            resize_factor = 3

            image = cv2.resize(inp_image, (0, 0), fx=resize_factor, fy=resize_factor)
            logger.debug(
                f"{LP} Resize factor: {resize_factor} - Original shape: {inp_image.shape} "
                f"-- Resized shape: {image.shape}"
            )

            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.ocr_cfg.debug_images:
                cv2.imwrite(
                    (self.ocr_cfg.dbg_img_dir / f"0{num}_02-Gray.jpg").as_posix(),
                    img_gray,
                )
            dark_cfg = self.ocr_cfg.darkness
            if dark_cfg:
                dark_thresh = dark_cfg.dark_thresh
                if dark_thresh is None:
                    dark_thresh = 100
                if dark_cfg.enabled:
                    if self.is_dark_image(img_gray, dark_thresh):
                        logger.debug(
                            f"{LP} Image is dark based on a threshold of {dark_thresh}/255, "
                            f"performing histogram stretching and auto adjusting brightness/contrast"
                        )
                        img_gray = await self.stretch_histogram(img_gray)
                        # perform auto brightness and contrast adjustment
                        m_i = self.ocr_cfg.darkness.mean_intensity
                        con_ = self.ocr_cfg.darkness.contrast
                        img_gray = self.auto_adjust_brightness_contrast(
                            img_gray, m_i, con_
                        )
                        if self.ocr_cfg.debug_images:
                            cv2.imwrite(
                                (
                                    self.ocr_cfg.dbg_img_dir
                                    / f"0{num}_02-Gray-Hist_Stretch-Auto_Brighten.jpg"
                                ).as_posix(),
                                img_gray,
                            )

            blur_kern = self.ocr_cfg.blur_kernel
            dilate_kern = self.ocr_cfg.dilate_kernel
            erode_kern = self.ocr_cfg.erode_kernel
            if blur_kern is None:
                # do an adaptive kernel based on image size, it must be odd

                blur_kern = (int(image.shape[0] / 100), int(image.shape[1] / 100))
            if dilate_kern is None:
                # do an adaptive kernel based on image size
                dilate_kern = (int(image.shape[0] / 100), int(image.shape[1] / 100))
            if erode_kern is None:
                # do an adaptive kernel based on image size
                erode_kern = (int(image.shape[0] / 100), int(image.shape[1] / 100))
            blur_kern = list(blur_kern)
            dilate_kern = list(dilate_kern)
            erode_kern = list(erode_kern)

            if blur_kern[0] % 2 == 0:
                blur_kern[0] += 1
            if blur_kern[1] % 2 == 0:
                blur_kern[1] += 1
            if dilate_kern[0] % 2 == 0:
                dilate_kern[0] += 1
            if dilate_kern[1] % 2 == 0:
                dilate_kern[1] += 1
            if erode_kern[0] % 2 == 0:
                erode_kern[0] += 1
            if erode_kern[1] % 2 == 0:
                erode_kern[1] += 1

            logger.debug(
                f"blur_kern: {blur_kern} -- dilate_kern: {dilate_kern} -- erode_kern: {erode_kern}"
            )

            img_blur = cv2.GaussianBlur(img_gray, tuple(blur_kern), 3)
            img_to_thresh, i2t_name = img_blur, "BLUR"
            logger.debug(
                f"Input to thresholding is '{i2t_name}'->thresh->dilate->erode"
            )

            img_thresh = cv2.threshold(
                img_to_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
            img_dila = cv2.dilate(img_thresh, tuple(dilate_kern), iterations=2)
            img_erode = cv2.erode(img_dila, tuple(erode_kern), iterations=2)

            processed_img = img_erode

            lp_canny_thresh = self.ocr_cfg.lp_edge_thresh
            if lp_canny_thresh is None or not isinstance(lp_canny_thresh, tuple):
                lp_canny_thresh = (110, 200)
            char_canny_thresh = self.ocr_cfg.char_edge_thresh
            if char_canny_thresh is None or not isinstance(char_canny_thresh, tuple):
                char_canny_thresh = (110, 200)

            img_lp_canny = cv2.Canny(
                processed_img, lp_canny_thresh[0], lp_canny_thresh[1]
            )
            img_char_canny = cv2.Canny(
                processed_img, char_canny_thresh[0], char_canny_thresh[1]
            )
            if self.ocr_cfg.debug_images:
                cv2.imwrite(
                    (
                        self.ocr_cfg.dbg_img_dir / f"0{self._num}_07-Canny-PLATE.jpg"
                    ).as_posix(),
                    img_lp_canny,
                )
                cv2.imwrite(
                    (
                        self.ocr_cfg.dbg_img_dir / f"0{self._num}_07-Canny-CHAR.jpg"
                    ).as_posix(),
                    img_char_canny,
                )
            transformed_img = self.perspective_transform(img_lp_canny, processed_img)
            letters_transform = []
            if transformed_img is None:
                logger.debug(f"{LP} No perspective transform performed")
            else:
                img_char_canny = cv2.Canny(
                    transformed_img, char_canny_thresh[0], char_canny_thresh[1]
                )
                letters_transform = self.char_extract_from_image(
                    img_char_canny, transformed_img
                )

            # perform character segmentation
            letters_proc = self.char_extract_from_image(img_char_canny, processed_img)

            imgs = []
            # 0 will be OTSU-threshold image with perspective transform (if able to)
            imgs.append(processed_img)
            imgs.append(img_gray)
            imgs.append(img_blur)
            imgs.append(img_thresh)
            imgs.append(img_dila)
            imgs.append(img_erode)
            imgs.append(img_lp_canny)
            img_names = [
                "FINAL",
                "Gray",
                "Blur",
                "Threshold-OTSU",
                "Dilate",
                "Erode",
                "Plate-Edges",
            ]
            easy_letter2word = ""
            easy_l2w_confs = []
            paddle_letter2word = ""
            paddle_l2w_confs = []
            tesseract_letter2word = ""
            tesseract_l2w_confs = []
            easy_letter2word_transform = ""
            easy_l2w_confs_transform = []
            paddle_letter2word_transform = ""
            paddle_l2w_confs_transform = []
            tesseract_letter2word_transform = ""
            tesseract_l2w_confs_transform = []

            for ocr_engine in self.ocr_readers:
                if isinstance(ocr_engine, easyocr.Reader):
                    cfg = self.ocr_cfg.easy_ocr
                    es = time.time()
                    kwargs = {
                        # "decoder": "greedy",
                        # "beamWidth": 5,
                        "width_ths": cfg.width_ths,
                        "height_ths": cfg.height_ths,
                        "slope_ths": cfg.slope_ths,
                        "ycenter_ths": cfg.ycenter_ths,
                        "min_size": cfg.min_size,
                        "contrast_ths": cfg.contrast_ths,
                        "adjust_contrast": cfg.adjust_contrast,
                        # "paragraph": True,
                    }
                    for idx, _img in enumerate(imgs):
                        kwargs["image"] = _img
                        task = await loop.run_in_executor(
                            None, functools.partial(ocr_engine.readtext, **kwargs)
                        )
                        ret_[f"EasyOCR_FULL-IMAGE_{img_names[idx]}"] = task

                    for idx, ltr in enumerate(letters_proc):
                        # None uses the default executor (ThreadPoolExecutor)
                        kwargs["image"] = ltr
                        task = await loop.run_in_executor(
                            None, functools.partial(ocr_engine.readtext, **kwargs)
                        )
                        ret_[f"EasyOCR_LETTER_{idx}"] = task
                        # [([[1,2], [3, 4]], 'letters', 0.56), ()]
                        if task:
                            easy_letter2word += task[0][1]
                            easy_l2w_confs.append(task[0][2])

                    for idx, ltr in enumerate(letters_transform):
                        # None uses the default executor (ThreadPoolExecutor)
                        kwargs["image"] = ltr
                        task = await loop.run_in_executor(
                            None, functools.partial(ocr_engine.readtext, **kwargs)
                        )
                        ret_[f"EasyOCR_LETTER_TRANSFORM_{idx}"] = task
                        if task:
                            easy_letter2word_transform += task[0][1]
                            easy_l2w_confs_transform.append(task[0][2])
                    logger.debug(
                        f"perf:{self.LP} EasyOCR took {time.time() - es:.5f} seconds for {len(letters_proc)+len(imgs)} images"
                    )

                if PaddleOCR is not None:
                    if isinstance(ocr_engine, PaddleOCR):
                        cfg = self.ocr_cfg.paddle_ocr
                        ps = time.time()
                        for idx, _img in enumerate(imgs):
                            results = await loop.run_in_executor(
                                None, ocr_engine.ocr, _img
                            )
                            ret_[f"PaddleOCR_FULL-IMAGE_{img_names[idx]}"] = results
                        for idx, ltr in enumerate(letters_proc):
                            results = await loop.run_in_executor(
                                None, ocr_engine.ocr, ltr
                            )
                            ret_[f"PaddleOCR_LETTER_{idx}"] = results
                            # [
                            #   [
                            #     [[[1.0, 2.0], [3.0, 4.0]], ('letters', 0.59)],
                            #     [[[5.0, 6.0], [7.0, 8.0]], ('letters', 0.69)]
                            #   ]
                            # ]
                            if results:
                                paddle_letter2word += results[0][0][1][0]
                                paddle_l2w_confs.append(results[0][0][1][1])
                        logger.debug(
                            f"perf:{self.LP} PaddleOCR took {time.time() - ps:.5f}  for {len(letters_proc)+len(imgs)} images seconds"
                        )
                        for idx, ltr in enumerate(letters_transform):
                            results = await loop.run_in_executor(
                                None, ocr_engine.ocr, ltr
                            )
                            ret_[f"PaddleOCR_LETTER_TRANSFORM_{idx}"] = results
                            if results:
                                paddle_letter2word_transform += results[0][0][1][0]
                                paddle_l2w_confs_transform.append(results[0][0][1][1])
                        # try:
                        #     boxes = [line[0] for line in results if line]
                        #     txts = [line[1][0] for line in results if line]
                        #     scores = [line[1][1] for line in results if line]
                        #     im_show = draw_ocr(image, boxes, txts, scores, font_path='/shared/simfang.ttf')
                        # except Exception as exc:
                        #     logger.error(f"\n\nError drawing OCR: {exc}")
                        # else:
                        #     if im_show is not None:
                        #         if self.ocr_config.debug_images:
                        #             cv2.imwrite((self.ocr_config.dbg_img_dir / f"0{self._num}_22-paddle.jpg").as_posix(), im_show)

                if pytesseract and ocr_engine == pytesseract:
                    cfg = self.ocr_cfg.tesseract
                    ts = time.time()
                    # TODO: try other psm to see if things improve
                    img_psm = cfg.img_psm
                    if img_psm is None:
                        img_psm = 7
                    char_psm = cfg.char_psm
                    if char_psm is None:
                        char_psm = 10

                    oem = cfg.oem
                    if oem is None:
                        oem = 3

                    lang = cfg.lang
                    if not lang:
                        lang = "eng"
                    kwargs = {
                        "lang": lang,
                        "config": f"--psm {img_psm} --oem {oem}",
                    }
                    try:
                        for idx, _img in enumerate(imgs):
                            kwargs["image"] = _img
                            task = await loop.run_in_executor(
                                None,
                                functools.partial(
                                    pytesseract.image_to_string,
                                    **kwargs,
                                ),
                            )
                            ret_[f"Tesseract_FULL-IMAGE_{img_names[idx]}"] = task

                        for idx, letter in enumerate(letters_proc):
                            kwargs["image"] = letter
                            if idx == 7:
                                # switch to treating each image as a single char
                                kwargs["config"] = f"--psm {char_psm} --oem {oem}"
                            task = await loop.run_in_executor(
                                None,
                                functools.partial(
                                    pytesseract.image_to_string,
                                    **kwargs,
                                ),
                            )
                            ret_[f"Tesseract_LETTER_{idx}"] = task
                            if task:
                                tesseract_letter2word += task
                                tesseract_l2w_confs.append(0.70)

                        for idx, letter in enumerate(letters_transform):
                            kwargs["image"] = letter
                            if idx == 7:
                                # switch to treating each image as a single char
                                kwargs["config"] = f"--psm {char_psm} --oem {oem}"
                            task = await loop.run_in_executor(
                                None,
                                functools.partial(
                                    pytesseract.image_to_string,
                                    **kwargs,
                                ),
                            )
                            ret_[f"Tesseract_LETTER_TRANSFORM_{idx}"] = task
                            if task:
                                tesseract_letter2word_transform += task
                                tesseract_l2w_confs_transform.append(0.70)

                    except Exception as exc:
                        logger.error(f"PYTESSERACT OCR ERROR: {exc}")
                    else:
                        logger.debug(
                            f"perf:{self.LP} PyTesseract took {time.time() - ts:.5f} s "
                            f"for {len(letters_proc)+len(imgs)} images"
                        )

        logger.debug(f"perf:{self.LP} All OCR took {time.time() - s:.5f} seconds")
        if self.ocr_cfg.debug_images:
            cv2.imwrite(
                (self.ocr_cfg.dbg_img_dir / f"0{num}_03-Blur.jpg").as_posix(),
                img_blur,
            )
            cv2.imwrite(
                (
                    self.ocr_cfg.dbg_img_dir / f"0{num}_04-Thresh_binary-otsu.jpg"
                ).as_posix(),
                img_thresh,
            )
            cv2.imwrite(
                (self.ocr_cfg.dbg_img_dir / f"0{num}_05-Dilate.jpg").as_posix(),
                img_dila,
            )
            cv2.imwrite(
                (self.ocr_cfg.dbg_img_dir / f"0{num}_06-Erode.jpg").as_posix(),
                img_erode,
            )

        # Change letters into words
        if easy_letter2word:
            avg = sum(easy_l2w_confs) / len(easy_l2w_confs)
            # [([[1,2], [3, 4]], 'letters', 0.56), ()]
            r = [
                (
                    [[0, 0], [0, 0]],
                    easy_letter2word,
                    avg,
                )
            ]

            ret_["EasyOCR_LETTER_TO_WORD"] = r
        if easy_letter2word_transform:
            avg = sum(easy_l2w_confs_transform) / len(easy_l2w_confs_transform)
            # [([[1,2], [3, 4]], 'letters', 0.56), ()]
            r = [
                (
                    [[0, 0], [0, 0]],
                    easy_letter2word_transform,
                    avg,
                )
            ]

            ret_["EasyOCR_LETTER_TO_WORD_TRANSFORM"] = r
        if paddle_letter2word:
            avg = sum(paddle_l2w_confs) / len(paddle_l2w_confs)
            # [
            #   [
            #     [[[1.0, 2.0], [3.0, 4.0]], ('letters', 0.59)],
            #     [[[5.0, 6.0], [7.0, 8.0]], ('letters', 0.69)]
            #   ]
            # ]
            r = [[[[[0, 0], [0, 0]], (paddle_letter2word, avg)]]]
            ret_["PaddleOCR_LETTER_TO_WORD"] = r
        if paddle_letter2word_transform:
            avg = sum(paddle_l2w_confs_transform) / len(paddle_l2w_confs_transform)
            r = [[[[[0, 0], [0, 0]], (paddle_letter2word_transform, avg)]]]
            ret_["PaddleOCR_LETTER_TO_WORD_TRANSFORM"] = r
        if tesseract_letter2word:
            avg = sum(tesseract_l2w_confs) / len(tesseract_l2w_confs)
            r = [[[[0, 0], [0, 0]], tesseract_letter2word, avg]]
            ret_["Tesseract_LETTER_TO_WORD"] = r
        if tesseract_letter2word_transform:
            avg = sum(tesseract_l2w_confs_transform) / len(
                tesseract_l2w_confs_transform
            )
            r = [[[[0, 0], [0, 0]], tesseract_letter2word_transform, avg]]
            ret_["Tesseract_LETTER_TO_WORD_TRANSFORM"] = r

        return ret_

    def char_extract_from_image(
        self, edged_img, extract_image: np.ndarray
    ) -> List[Optional[np.ndarray]]:
        """Supply a image that has been edge detected and get a list of cropped letter images

        :param edged_img: Processed and edge detected image
        :param extract_image: Image to extract letters from (Must be same dimensions)
        """
        lp: str = "extract chars:"
        cfg: Optional[OCRConfig.CharExtractConfig] = None
        if self.ocr_cfg.char_extract:
            cfg = self.ocr_cfg.char_extract
        if not cfg:
            cfg = OCRConfig.CharExtractConfig()

        enabled = cfg.enabled
        if enabled is None:
            enabled = True

        if enabled is False:
            logger.warning(f"{lp} Char extraction is disabled, returning empty list")
            return []

        contours, hierarchy = cv2.findContours(
            edged_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            logger.debug(f"{lp} No contours found")
            return []

        letters = []
        height, width = edged_img.shape[:2]
        img_copy = extract_image.copy()
        img_area = height * width

        min_area_perc = cfg.min_area_perc
        if min_area_perc is None:
            min_area_perc = 0.02

        max_area_perc = cfg.max_area_perc
        if max_area_perc is None:
            max_area_perc = 0.6

        max_w_perc = cfg.max_w_perc
        if max_w_perc is None:
            max_w_perc = 0.3875

        min_w_perc = cfg.min_w_perc
        if min_w_perc is None:
            min_w_perc = 0.025

        max_h_perc = cfg.max_h_perc
        if max_h_perc is None:
            max_h_perc = 0.9

        min_h_perc = cfg.min_h_perc
        if min_h_perc is None:
            min_h_perc = 0.2

        leeway_w_perc = cfg.leeway_w_perc
        if leeway_w_perc is None:
            leeway_w_perc = 0.01
        leeway_h_perc = cfg.leeway_h_perc
        if leeway_h_perc is None:
            leeway_h_perc = 0.01
        max_aspect_ratio = cfg.max_aspect_ratio
        if max_aspect_ratio is None:
            max_aspect_ratio = 0.25
        min_aspect_ratio = cfg.min_aspect_ratio
        if min_aspect_ratio is None:
            min_aspect_ratio = 0.05
        contour_scale = cfg.contour_approx_scale
        if contour_scale is None:
            contour_scale = 0.02

        # normalized
        min_area = min_area_perc * img_area
        max_area = max_area_perc * img_area
        min_h = max_h_perc * height
        max_h = max_h_perc * height
        min_w = min_w_perc * width
        max_w = max_w_perc * width
        leeway_w = int(width * leeway_w_perc)
        leeway_h = int(height * leeway_h_perc)

        hits = []
        processed_indices = set()

        for idx, contour in enumerate(contours):
            # Check if the contour has already been processed (child contour that was stacked)
            if idx in processed_indices:
                logger.debug(
                    f"{lp} Contour {idx} has already been processed, skipping..."
                )
                continue
            # Check if the contour has a parent (child contours exist)
            if hierarchy[0][idx][3] == -1:
                pass
            else:
                logger.debug(
                    f"{lp} Contour {idx} has a parent, it is a child contour, processing into a single contour..."
                )
                # Contour is a child, consider combining it with parent contours
                # Iterate through hierarchy to find other child contours of the same parent
                child_contours = [idx]
                current_child = hierarchy[0][idx][0]
                while current_child != -1:
                    child_contours.append(current_child)
                    processed_indices.add(current_child)
                    current_child = hierarchy[0][current_child][0]
                # Combine child contours with the parent and process the combined contour
                logger.debug(f"{lp} Combining {len(child_contours)} contours...")
                contour = np.vstack([contours[child] for child in child_contours])

            area = cv2.contourArea(contour)
            norm_area = area / img_area
            if area < min_area:
                logger.debug(
                    f"{lp} Contour {idx} area: {area} ({norm_area*100:.3f}) is < "
                    f"{min_area} ({min_area_perc * 100:.3f}%) of image, - (too small of a blob "
                    f"to be a char) - skipping..."
                )
                continue
            if area > max_area:
                logger.debug(
                    f"{lp} Contour {idx} area: {area} ({norm_area*100:.3f}) is > {max_area} ("
                    f"{max_area_perc * 100:.3f}%) of image - (too large of a blob to be a char) - "
                    f"skipping..."
                )
                continue

            peri = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, peri, True)
            if len(approx) != 4:
                logger.debug(f"{lp} Contour {idx} is not a quadrilateral, skipping...")
                continue
            # It is a quadrilateral, get rectangle coordinates
            (x, y, w, h) = cv2.boundingRect(contour)

            # normalize width and height
            norm_h = h / height
            norm_w = w / width

            if w < min_w:
                logger.debug(
                    f"{lp} Contour {idx} width {w} ({norm_w*100:.3f}) is < {min_w} ({min_w_perc*100:.3f}%) - (too skinny "
                    f"to be a char) - skipping..."
                )
                continue
            if w > max_w:
                # too wide to be a character
                logger.debug(
                    f"{lp} Contour {idx} width {w} ({norm_w*100:.3f}) is > {max_w} ({max_w_perc*100:.3f}%) "
                    f" - (too wide to be a char) - skipping..."
                )
                continue

            if h < min_h:
                logger.debug(
                    f"{lp} Contour {idx} height {h} ({norm_h*100:.3f}) is < {min_h} ({min_h_perc*100:.3f}%) - "
                    f"(too short to be a char) - skipping..."
                )
                continue
            if h > max_h:
                logger.debug(
                    f"{lp} Contour {idx} height {h} ({norm_h*100:.3f}) is > {max_h} ({max_h_perc*100:.3f}%) "
                    f"- (too tall to be a char) - skipping..."
                )
                continue

            aspect_ratio = w / h
            logger.debug(f"{lp} Contour {idx} aspect ratio: {aspect_ratio}")

            logger.debug(
                f"{lp} Contour {idx} is a potential character (quadrilateral within area, w, h bounds)"
            )
            hits.append((x, y, w, h))

        # sort based on x position (left to right or right to left)
        right2left = self.ocr_cfg.char_extract.right2left
        if right2left is None:
            right2left = False

        hits.sort(key=lambda z: z[0], reverse=right2left)
        for idx, (x, y, w, h) in enumerate(hits):
            # add leeway pixels to the crop
            _y = y - leeway_h
            _h = h + leeway_h * 2
            _x = x - leeway_w
            _w = w + leeway_w * 2
            # extract letters from processed image (not edge detected one)
            letter_crop = extract_image[_y : _y + _h, _x : _x + _w]
            # draw rects if debug image
            if letter_crop is not None and letter_crop.any():
                if self.ocr_cfg.debug_images:
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255))
                    cv2.imwrite(
                        (
                            self.ocr_cfg.dbg_img_dir
                            / f"0{self._num}_99-letter_{idx+1}.jpg"
                        )
                        .expanduser()
                        .resolve()
                        .as_posix(),
                        letter_crop,
                    )

                letters.append(letter_crop)

        if self.ocr_cfg.debug_images:
            cv2.imwrite(
                (self.ocr_cfg.dbg_img_dir / f"0{self._num}_10-Segment_Chars.jpg")
                .expanduser()
                .resolve()
                .as_posix(),
                img_copy,
            )
        logger.debug(f"{lp} Found {len(letters)} POSSIBLE characters")
        return letters

    def plate_recognizer(self, image: np.ndarray) -> Optional[List[Tuple]]:
        s = time.time()
        result = None

        logger.debug(f"perf:Plate Recognizer: OCR took {time.time() - s:.5f} seconds")
        return result

    def openalpr(self, image: np.ndarray) -> Optional[List[Tuple]]:
        s = time.time()
        result = None

        logger.debug(f"perf:OpenALPR: OCR took {time.time() - s:.5f} seconds")
        return result

    def decode_cv_east_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < 0.2:
                    continue
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

    def order_points(self, pts):
        # Step 1: Find centre of object
        center = np.mean(pts)

        # Step 2: Move coordinate system to centre of object
        shifted = pts - center

        # Step #3: Find angles subtended from centroid to each corner point
        theta = np.arctan2(shifted[:, 0], shifted[:, 1])

        # Step #4: Return vertices ordered by theta
        ind = np.argsort(theta)
        return pts[ind]

    def find_warp_plate(self, img: np.ndarray, orig: np.ndarray):
        """Look for a plate (large contour) in the image and warp for a perspective transform.

        The input image has already had edge detection and thresholding applied.
        """
        lp: str = "find_warp_plate:"

        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            logger.debug(f"{lp} No contours found")
            return None, None, None

        logger.debug(f"{lp} {len(contours)} contours found")
        cfg = self.ocr_cfg.warp_plate
        if not cfg:
            cfg = OCRConfig.WarpPlateConfig()

        enabled = cfg.enabled
        if enabled is None:
            enabled = False

        if enabled is False:
            logger.warning(f"{lp} License plate warping is disabled, skipping...")
            return None, None, None

        biggest = np.array([])
        largest_area_so_far = 0
        index = None
        img_contour = None
        height, width = orig.shape[:2]
        img_area = height * width

        min_area_perc = cfg.min_area_perc
        if min_area_perc is None:
            min_area_perc = 0.02

        max_area_perc = cfg.max_area_perc
        if max_area_perc is None:
            max_area_perc = 0.6

        max_w_perc = cfg.max_w_perc
        if max_w_perc is None:
            max_w_perc = 0.3875

        min_w_perc = cfg.min_w_perc
        if min_w_perc is None:
            min_w_perc = 0.025

        max_h_perc = cfg.max_h_perc
        if max_h_perc is None:
            max_h_perc = 0.9

        min_h_perc = cfg.min_h_perc
        if min_h_perc is None:
            min_h_perc = 0.2

        max_aspect_ratio = cfg.max_aspect_ratio
        if max_aspect_ratio is None:
            max_aspect_ratio = 0.25
        min_aspect_ratio = cfg.min_aspect_ratio
        if min_aspect_ratio is None:
            min_aspect_ratio = 0.05
        contour_scale = cfg.contour_approx_scale
        if contour_scale is None:
            contour_scale = 0.02

        # normalized
        min_area = min_area_perc * img_area
        max_area = max_area_perc * img_area
        min_h = max_h_perc * height
        max_h = max_h_perc * height
        min_w = min_w_perc * width
        max_w = max_w_perc * width

        img_aspect_ratio = width / height

        logger.debug(
            f"{lp} {img_area=} -- {img_aspect_ratio=} -- Contour_Approximation_Scale={contour_scale}"
        )

        processed_indices = set()  # Set to keep track of processed contours

        for i, contour in enumerate(contours):
            # Check if the contour has already been processed (child contour that was stacked)
            if i in processed_indices:
                logger.debug(
                    f"{lp} Contour {i} has already been processed, skipping..."
                )
                continue
            # Check if the contour has a parent (child contours exist)
            if hierarchy[0][i][3] == -1:
                pass
            else:
                logger.debug(
                    f"{lp} Contour {i} has a parent, it is a child contour, processing into a single contour..."
                )
                # Contour is a child, consider combining it with parent contours
                # Iterate through hierarchy to find other child contours of the same parent
                child_contours = [i]
                current_child = hierarchy[0][i][0]
                while current_child != -1:
                    child_contours.append(current_child)
                    processed_indices.add(current_child)
                    current_child = hierarchy[0][current_child][0]
                # Combine child contours with the parent and process the combined contour
                logger.debug(f"{lp} Combining {len(child_contours)} contours...")
                contour = np.vstack([contours[child] for child in child_contours])

            area = cv2.contourArea(contour)
            norm_area = area / img_area
            if area >= min_area:
                logger.debug(
                    f"{lp} Contour {i} area: {area} ({norm_area*100:.3f}) is >= {min_area} ({max_area_perc*100}%) "
                    f"of image, performing perspective transform..."
                )

                peri = contour_scale * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, peri, True)
                if area >= largest_area_so_far and len(approx) == 4:
                    # len(approx) == 4 means the contour has 4 points (rect)
                    biggest = approx
                    largest_area_so_far = area
                    index = i
            # else:
            #     logger.debug(
            #         f"{lp} Contour {i} area: {area} px ({(area/img_area)*100:.2f}) is < {min_area} ({round(perc*100)}% "
            #         f"of image, skipping..."
            #     )

        warped = None
        if index is not None:
            logger.debug(f"{lp} performing perspective transform on contour {index}...")
            img_contour = orig.copy()
            cv2.drawContours(img_contour, contours, index, (255, 0, 0), 3)
            src = np.squeeze(biggest).astype(np.float32)  # Source points
            # Destination points
            dst = np.float32(
                [[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]]
            )
            # Order the points correctly
            biggest = self.order_points(src)
            dst = self.order_points(dst)
            # Get the perspective transform
            transform_matrix = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(
                orig, transform_matrix, (height, width), flags=cv2.INTER_LINEAR
            )
            logger.debug(
                f"Original image shape = {orig.shape}, HEIGHT = {height}, WIDTH = {width}\n"
                f"Warped image shape = {warped.shape}, HEIGHT = {warped.shape[0]}, WIDTH = {warped.shape[1]}\n"
            )
            # if the warped image does not have same dimensions as the original image, ascertain which way to rotate it
            if warped.shape[:2] != orig.shape[:2]:
                logger.debug(
                    f"{lp} Warped image shape does not match source, rotating 90 degrees clockwise..."
                )
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        return biggest, img_contour, warped

    def check_similarity(
        self, word1, word2, thresh: Optional[int] = None
    ) -> Optional[bool]:
        """Uses Levenshtein distance to check if two words are similar.
        If it returns None, the python-Levenshtein package is not installed.
        """
        if Levenshtein is not None:
            # Calculate the Levenshtein distance between the two words
            distance = Levenshtein.distance(word1, word2)

            # Define a threshold for similarity
            similarity_threshold = thresh
            if similarity_threshold is None:
                similarity_threshold = 3

            # Check if the Levenshtein distance is below the threshold
            if distance <= similarity_threshold:
                return True
        else:
            logger.warning(f"python-Levenshtein package not installed!")
            return None

        return False

    def perspective_transform(
        self, image: np.ndarray, proc_img: np.ndarray
    ) -> Optional[np.ndarray]:
        """Image processing for the license plate, perform dynamic perspective transformation.

        Input image is already processed.

        :param image: License plate image, already processed and edge detected
        :param proc_img: Processed image to extract letters from (resize, blur, thresh, dilate, erode)
        :return: Warped image | None
        """
        s = time.time()
        biggest_contour, img_contour, img_warped = self.find_warp_plate(image, proc_img)
        logger.debug(
            f"perf:{self.LP} Looking for plate contours and perspective transform "
            f"took {time.time() - s:.5f} s"
        )
        if img_contour is not None and self.ocr_cfg.debug_images:
            cv2.imwrite(
                (self.ocr_cfg.dbg_img_dir / f"0{self._num}_08-Contour.jpg").as_posix(),
                img_contour,
            )
        if img_warped is not None and self.ocr_cfg.debug_images:
            cv2.imwrite(
                (self.ocr_cfg.dbg_img_dir / f"0{self._num}_08-Warped.jpg").as_posix(),
                img_warped,
            )
        return img_warped
