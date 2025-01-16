from .config import *

MODEL_PATH = attach_home_root('model/HiNetUnet_shared_rev_blocks0/')
os.makedirs(MODEL_PATH, exist_ok=True)

IMAGE_PATH = attach_home_root('show/HiNetUnet_shared_rev_blocks0/')
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'
IMAGE_PATH_cats = IMAGE_PATH + 'cats/'
os.makedirs(IMAGE_PATH_cover, exist_ok=True)
os.makedirs(IMAGE_PATH_secret, exist_ok=True)
os.makedirs(IMAGE_PATH_steg, exist_ok=True)
os.makedirs(IMAGE_PATH_secret_rev, exist_ok=True)
os.makedirs(IMAGE_PATH_cats, exist_ok=True)

model_name = "HiNetUnet"

cropsize = 512
model_config = dict(shared_rev_blocks=False)