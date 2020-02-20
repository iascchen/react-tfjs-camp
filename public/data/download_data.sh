BASE_URL = https://storage.googleapis.com/cvdf-datasets/mnist

TRAIN_IMAGES_FILE = $BASE_URL/train-images-idx3-ubyte.gz
TRAIN_LABELS_FILE = $BASE_URL/train-labels-idx1-ubyte.gz
TEST_IMAGES_FILE = $BASE_URL/t10k-images-idx3-ubyte.gz
TEST_LABELS_FILE = $BASE_URL/t10k-labels-idx1-ubyte.gz

wget TRAIN_IMAGES_FILE
wget TRAIN_LABELS_FILE
wget TEST_IMAGES_FILE
wget TEST_LABELS_FILE
