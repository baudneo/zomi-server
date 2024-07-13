import logging


SERVER_LOGGER_NAME = "ML:API"
SERVER_LOG_FORMAT = logging.Formatter(
    "%(asctime)s.%(msecs)d %(name)s %(levelname)s[%(module)s:%(lineno)d]> %(message)s",
    "%m/%d/%y %H:%M:%S",
)