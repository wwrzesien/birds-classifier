################################ LIBRARIES START ###############################
import torch
import torch.nn as nn
################################# LIBRARIES END ################################

# CONSTANTS
TRAIN = 'train'
TEST = 'test'
VAL = 'valid'
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_GPU else "cpu") # "cuda:0" == "gpu" ?
PRESENTATION_BATCH = 8
SUMMARY_SHAPE = (3, 224, 224)

# MODEL PARAMETERS
LR = 0.001
MOMENTUM = 0.9
GAMMA = 0.1
CRITERION = nn.CrossEntropyLoss()

# TRAINING 
TRAIN_SHOW_BATCH_CHANGE = 10
BATCH_SIZE = 100
NUMBER_OF_EPOCHS = 50

# CSV FORMAT
# COLUMNS = ['EPOCH', 'avg_acc', 'avg_loss', 'avg_acc_val', 'avg_loss_val']
CSV_DELIMITER = ';'
CSV_NEWLINE = ''

# DATA LOADERS
NUM_WORKERS = 2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ZADANIE 1a
EX1a_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad1a/'
MODEL_PATH_EX1a = EX1a_FOLDER_PATH + 'VGG16_bird260_1a.pt'
HISTORY_PATH_EX1a = EX1a_FOLDER_PATH + 'history_1a.csv'
PNG_LOSS_PATH_EX1a = EX1a_FOLDER_PATH + 'loss_1a.png'
PNG_ACCURACY_PATH_EX1a = EX1a_FOLDER_PATH + 'accuracy_1a.png'
TRAINING_PARAMS_PATH_EX1a = EX1a_FOLDER_PATH + 'training_params_1a.txt'
# False - start training from scratch, True - use previous weights
RESUME_TRAINING_EX1a = False


# ZADANIE 1c LIN
EX1cLIN_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad1cLIN/'
MODEL_PATH_EX1cLIN = EX1cLIN_FOLDER_PATH + 'VGG16_bird260_1cLIN.pt'
HISTORY_PATH_EX1cLIN = EX1cLIN_FOLDER_PATH + 'history_1cLIN.csv'
PNG_LOSS_PATH_EX1cLIN = EX1cLIN_FOLDER_PATH + 'loss_1cLIN.png'
PNG_ACCURACY_PATH_EX1cLIN = EX1cLIN_FOLDER_PATH + 'accuracy_1cLIN.png'
TRAINING_PARAMS_PATH_EX1cLIN = EX1cLIN_FOLDER_PATH + 'training_params_1cLIN.txt'
RESUME_TRAINING_EX1cLIN = False


# ZADANIE 1c QUAD
EX1cQUAD_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad1cQUAD/'
MODEL_PATH_EX1cQUAD = EX1cQUAD_FOLDER_PATH + 'VGG16_bird260_1cQUAD.pt'
HISTORY_PATH_EX1cQUAD = EX1cQUAD_FOLDER_PATH + 'history_1cQUAD.csv'
PNG_LOSS_PATH_EX1cQUAD = EX1cQUAD_FOLDER_PATH + 'loss_1cQUAD.png'
PNG_ACCURACY_PATH_EX1cQUAD = EX1cQUAD_FOLDER_PATH + 'accuracy_1cQUAD.png'
TRAINING_PARAMS_PATH_EX1cQUAD = EX1cQUAD_FOLDER_PATH + 'training_params_1cQUAD.txt'
RESUME_TRAINING_EX1cQUAD = False


# ZADANIE 1c RBF
EX1cRBF_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad1cRBF/'
MODEL_PATH_EX1cRBF = EX1cRBF_FOLDER_PATH + 'VGG16_bird260_1cRBF.pt'
HISTORY_PATH_EX1cRBF = EX1cRBF_FOLDER_PATH + 'history_1cRBF.csv'
PNG_LOSS_PATH_EX1cRBF = EX1cRBF_FOLDER_PATH + 'loss_1cRBF.png'
PNG_ACCURACY_PATH_EX1cRBF = EX1cRBF_FOLDER_PATH + 'accuracy_1cRBF.png'
TRAINING_PARAMS_PATH_EX1cRBF = EX1cRBF_FOLDER_PATH + 'training_params_1cRBF.txt'
RESUME_TRAINING_EX1cRBF = False

# ZADANIE 2a
EX2a_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad2a/'
MODEL_PATH_EX2a = EX2a_FOLDER_PATH + 'VGG16_bird260_2a.pt'
HISTORY_PATH_EX2a = EX2a_FOLDER_PATH + 'history_2a.csv'
PNG_LOSS_PATH_EX2a = EX2a_FOLDER_PATH + 'loss_2a.png'
PNG_ACCURACY_PATH_EX2a = EX2a_FOLDER_PATH + 'accuracy_2a.png'
TRAINING_PARAMS_PATH_EX2a = EX2a_FOLDER_PATH + 'training_params_2a.txt'
RESUME_TRAINING_EX2a = False

# ZADANIE 2b
EX2b_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad2b/'
MODEL_PATH_EX2b = EX2b_FOLDER_PATH + 'VGG16_bird260_2b.pt'
HISTORY_PATH_EX2b = EX2b_FOLDER_PATH + 'history_2b.csv'
PNG_LOSS_PATH_EX2b = EX2b_FOLDER_PATH + 'loss_2b.png'
PNG_ACCURACY_PATH_EX2b = EX2b_FOLDER_PATH + 'accuracy_2b.png'
TRAINING_PARAMS_PATH_EX2b = EX2b_FOLDER_PATH + 'training_params_2b.txt'
RESUME_TRAINING_EX2b = False

# ZADANIE 2c
EX2c_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad2c/'
MODEL_PATH_EX2c = EX2c_FOLDER_PATH + 'VGG16_bird260_2c.pt'
HISTORY_PATH_EX2c = EX2c_FOLDER_PATH + 'history_2c.csv'
PNG_LOSS_PATH_EX2c = EX2c_FOLDER_PATH + 'loss_2c.png'
PNG_ACCURACY_PATH_EX2c = EX2c_FOLDER_PATH + 'accuracy_2c.png'
TRAINING_PARAMS_PATH_EX2c = EX2c_FOLDER_PATH + 'training_params_2c.txt'
RESUME_TRAINING_EX2c = False

# ZADANIE 2d
EX2d_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad2d/'
MODEL_PATH_EX2d = EX2d_FOLDER_PATH + 'VGG16_bird260_2d.pt'
HISTORY_PATH_EX2d = EX2d_FOLDER_PATH + 'history_2d.csv'
PNG_LOSS_PATH_EX2d = EX2d_FOLDER_PATH + 'loss_2d.png'
PNG_ACCURACY_PATH_EX2d = EX2d_FOLDER_PATH + 'accuracy_2d.png'
TRAINING_PARAMS_PATH_EX2d = EX2d_FOLDER_PATH + 'training_params_2d.txt'
RESUME_TRAINING_EX2d = False

# ZADANIE 2e
EX2e_FOLDER_PATH = '/content/drive/MyDrive/260_Bird_Species/Model_weights/Zad2e/'
PNG_LOSS_PATH_EX2e = EX2e_FOLDER_PATH + 'loss_2e'
PNG_ACCURACY_PATH_EX2e = EX2e_FOLDER_PATH + 'accuracy_2e'

