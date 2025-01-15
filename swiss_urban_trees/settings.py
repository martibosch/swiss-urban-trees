"""Settings."""

import logging as lg

# utils
# REQUEST_KWS = {}
# PAUSE = 1
# ERROR_PAUSE = 60
TIMEOUT = 180
MAX_RETRIES = 5

# logging
LOG_CONSOLE = False
LOG_FILE = False
LOG_FILENAME = "swiss_urban_trees"
LOG_LEVEL = lg.INFO
LOG_NAME = "swiss_urban_trees"
LOGS_FOLDER = "./logs"

# WMS settings
RES = 0.2
MAX_SIZE = 1000
SCALE_FACTOR = 1
FORMAT = "image/png"
NODATA = 255
