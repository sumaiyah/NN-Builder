DATASET_NAME = 'LetterRecognition-binary'

# Configure file paths and initialise files
BASE_PATH = 'C:/Users/sumaiyah/OneDrive - University Of Cambridge/Project/data/%s/' % DATASET_NAME
MODEL_PATH = BASE_PATH + 'models/'
LABEL_PATH = BASE_PATH + 'labels/'

DATA_PATH = BASE_PATH + 'data.csv'
FOLD_PATH = BASE_PATH + 'fold_indices.txt'

NN_INFORMATION_PATH = BASE_PATH + 'information.txt'

NN_LABELS_PATH = LABEL_PATH + 'NN_labels.txt'
TRUE_LABELS_PATH = LABEL_PATH + 'TRUE_labels.txt'