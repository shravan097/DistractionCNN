import tensorflow as tf
# Dataset Parameters - CHANGE HERE
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
# the dataset file or root folder path.
DATASET_PATH = "/Users/shravan/Desktop/DistractionCNN/state-farm-distracted-driver-detection/imgs/train"
TEST_PATH = "/Users/shravan/Desktop/DistractionCNN/state-farm-distracted-driver-detection/imgs/test"
# Image Parameters
N_CLASSES = 10 # CHANGE HERE, total number of classes
IMG_HEIGHT = 200 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 200 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale
BATCH_SIZE = 128
imgClassDict = dict({
	"c0": "safe_driving",
	"c1": "texting_right",
	"c2": "talking_on_phone_right",
	"c3": "texting_left",
	"c4": "talking_on_phone_left",
	"c5": "operating_the_radio",
	"c6": "drinking",
	"c7": "reaching_behind",
	"c8": "hair_makeup",
	"c9": "talking_to_passenger",
})
LABELS = [0,1,2,3,4,5,6,7,8,9]
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_SIZE = None
VALIDATION_SET_SIZE = 2000