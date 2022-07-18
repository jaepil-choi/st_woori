from pathlib import Path

MIN_DATE = 20170101

class PathConfig:
    PROJ_PATH = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJ_PATH / 'data'
    ASSETS_PATH = PROJ_PATH / 'assets'
    APP_PATH = PROJ_PATH / 'app'