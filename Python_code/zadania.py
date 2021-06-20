################################ LIBRARIES START ###############################
# External Libraries
from joblib import dump, load
from importlib import reload
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchsummary
from torchsummary import summary
import os
from sklearn.svm import SVC

# Internal Libraries
import model_functions
model_functions = reload(model_functions)
from model_functions import eval_model, train_model, visualize_model, load_weights, \
  train_svm, eval_svm

import parameters
parameters = reload(parameters)
from parameters import USE_GPU, NUMBER_OF_EPOCHS, CRITERION, PRESENTATION_BATCH, \
    LR, MOMENTUM, GAMMA, SUMMARY_SHAPE, \
    EX1a_FOLDER_PATH, RESUME_TRAINING_EX1a, MODEL_PATH_EX1a, HISTORY_PATH_EX1a, \
    TRAINING_PARAMS_PATH_EX1a, PNG_LOSS_PATH_EX1a, PNG_ACCURACY_PATH_EX1a, \
    EX1cLIN_FOLDER_PATH, RESUME_TRAINING_EX1cLIN, MODEL_PATH_EX1cLIN, HISTORY_PATH_EX1cLIN, \
    TRAINING_PARAMS_PATH_EX1cLIN, PNG_LOSS_PATH_EX1cLIN, PNG_ACCURACY_PATH_EX1cLIN, \
    EX1cQUAD_FOLDER_PATH, RESUME_TRAINING_EX1cQUAD, MODEL_PATH_EX1cQUAD, HISTORY_PATH_EX1cQUAD, \
    TRAINING_PARAMS_PATH_EX1cQUAD, PNG_LOSS_PATH_EX1cQUAD, PNG_ACCURACY_PATH_EX1cQUAD, \
    EX1cRBF_FOLDER_PATH, RESUME_TRAINING_EX1cRBF, MODEL_PATH_EX1cRBF, HISTORY_PATH_EX1cRBF, \
    TRAINING_PARAMS_PATH_EX1cRBF, PNG_LOSS_PATH_EX1cRBF, PNG_ACCURACY_PATH_EX1cRBF, \
    EX2a_FOLDER_PATH, RESUME_TRAINING_EX2a, MODEL_PATH_EX2a, HISTORY_PATH_EX2a, \
    TRAINING_PARAMS_PATH_EX2a, PNG_LOSS_PATH_EX2a, PNG_ACCURACY_PATH_EX2a, \
    EX2b_FOLDER_PATH, RESUME_TRAINING_EX2b, MODEL_PATH_EX2b, HISTORY_PATH_EX2b, \
    TRAINING_PARAMS_PATH_EX2b, PNG_LOSS_PATH_EX2b, PNG_ACCURACY_PATH_EX2b, \
    EX2c_FOLDER_PATH, RESUME_TRAINING_EX2c, MODEL_PATH_EX2c, HISTORY_PATH_EX2c, \
    TRAINING_PARAMS_PATH_EX2c, PNG_LOSS_PATH_EX2c, PNG_ACCURACY_PATH_EX2c, \
    EX2d_FOLDER_PATH, RESUME_TRAINING_EX2d, MODEL_PATH_EX2d, HISTORY_PATH_EX2d, \
    TRAINING_PARAMS_PATH_EX2d, PNG_LOSS_PATH_EX2d, PNG_ACCURACY_PATH_EX2d, \
    EX2e_FOLDER_PATH, PNG_LOSS_PATH_EX2e, PNG_ACCURACY_PATH_EX2e


import models
models = reload(models)
from models import create_model_ex1a, create_model_ex1c, create_model_ex2a, \
  create_model_ex2b, create_model_ex2c, create_model_ex2d

import post_processing
post_processing = reload(post_processing)
from post_processing import save_training_params, plot_training, plot_compare_models

################################# LIBRARIES END ################################

#############
# ZADANIE 1 #
#############
# 1a)
# Zastosować wstępnie wytrenowaną sieć do uczenia tylko części klasyfikującej
# (ostatnie warstwy o połączeniach kompletnych)

def zadanie1a(class_names, dataloaders, dataset_sizes):

    try:
      os.mkdir(EX1a_FOLDER_PATH)
    except:
      print(EX1a_FOLDER_PATH + ' already exists')

    save_training_params(TRAINING_PARAMS_PATH_EX1a)

    VGG_16 = create_model_ex1a(class_names)

    VGG_16 = load_weights(VGG_16, MODEL_PATH_EX1a, resume_training = RESUME_TRAINING_EX1a)

    VGG_16.cuda()
    summary(VGG_16, SUMMARY_SHAPE)

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, VGG_16.parameters()), lr=LR, momentum = MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma = GAMMA)

    print("Test before training")
    eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
    visualize_model(VGG_16, dataloaders, class_names)
    VGG_16 = train_model(VGG_16, CRITERION, optimizer_ft, exp_lr_scheduler,
     dataloaders, dataset_sizes, HISTORY_PATH_EX1a, RESUME_TRAINING_EX1a, num_epochs=NUMBER_OF_EPOCHS)

    torch.save(VGG_16.state_dict(), MODEL_PATH_EX1a)

    plot_training(HISTORY_PATH_EX1a, PNG_ACCURACY_PATH_EX1a, PNG_LOSS_PATH_EX1a)

# 1b)
# Zanalizować wyniki klasyfikacji
def zadanie1b(class_names, dataloaders, dataset_sizes):
  
    VGG_16 = create_model_ex1a(class_names)

    resume_training = True

    VGG_16 = load_weights(VGG_16, MODEL_PATH_EX1a, resume_training = resume_training)

    print("Test after training")
    eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
    visualize_model(VGG_16, dataloaders, class_names, num_images = PRESENTATION_BATCH)

# 1c)
# Zastąpić część klasyfikującą sieci
# przez SVM dla jądra liniowego, kwadratowego i wykładniczego.
def zadanie1c_linearSVM(class_names, dataloaders, dataset_sizes):

  try:
    os.mkdir(EX1cLIN_FOLDER_PATH)
  except:
    print(EX1cLIN_FOLDER_PATH + ' already exists')

  save_training_params(TRAINING_PARAMS_PATH_EX1cLIN)

  VGG_16 = create_model_ex1c()
  
  VGG_16.cuda()
  print(VGG_16)
  summary(VGG_16, SUMMARY_SHAPE)

  linear_svc = SVC(kernel='linear', cache_size = 5000) # liniowy

  # TRAINING SVM
  linear_svc = train_svm(VGG_16, linear_svc, dataloaders)

  # dump(linear_svc, EX1cLIN_FOLDER_PATH + 'VGG16_bird260_1cLIN_max_iter_default.pt') 
  dump(linear_svc, MODEL_PATH_EX1cLIN)
  # load(linear_svc, MODEL_PATH_EX1cLIN) 


def zadanie1c_quadraticSVM(class_names, dataloaders, dataset_sizes):

  try:
    os.mkdir(EX1cQUAD_FOLDER_PATH)
  except:
    print(EX1cQUAD_FOLDER_PATH + ' already exists')

  save_training_params(TRAINING_PARAMS_PATH_EX1cQUAD)

  VGG_16 = create_model_ex1c()
  
  VGG_16.cuda()
  print(VGG_16)
  summary(VGG_16, SUMMARY_SHAPE)

  quadratic_svm = SVC(kernel='poly', degree = 2, cache_size = 10000) # kwadratowy

  # TRAINING SVM
  quadratic_svm = train_svm(VGG_16, quadratic_svm, dataloaders)
 
  dump(quadratic_svm, MODEL_PATH_EX1cQUAD)


def zadanie1c_rbfSVM(class_names, dataloaders, dataset_sizes):

  try:
    os.mkdir(EX1cRBF_FOLDER_PATH)
  except:
    print(EX1cRBF_FOLDER_PATH + ' already exists')

  save_training_params(TRAINING_PARAMS_PATH_EX1cRBF)

  VGG_16 = create_model_ex1c()
  
  VGG_16.cuda()
  print(VGG_16)
  summary(VGG_16, SUMMARY_SHAPE)

  rbf_svm = SVC(kernel='rbf', cache_size = 15000) # wykladniczy

  # TRAINING SVM
  rbf_svm = train_svm(VGG_16, rbf_svm, dataloaders)
 
  dump(rbf_svm, MODEL_PATH_EX1cRBF)

# 1d)
# Zanalizować wyniki klasyfikacji. W szczególności, zbadać efekt dopuszczenia
# błędnych klasyfikacji, porównać z wynikami 1a.
def zadanie1d_linearSVM(class_names, dataloaders, dataset_sizes):

  VGG_16 = create_model_ex1c()
  VGG_16.cuda()

  print("Linear kernel")
  linear_svm = load(MODEL_PATH_EX1cLIN)
  eval_svm(VGG_16, linear_svm, dataloaders, dataset_sizes)


def zadanie1d_quadraticSVM(class_names, dataloaders, dataset_sizes):

  VGG_16 = create_model_ex1c()
  VGG_16.cuda()

  print("Quadratic kernel")
  quadratic_svm = load(MODEL_PATH_EX1cQUAD)
  eval_svm(VGG_16, quadratic_svm, dataloaders, dataset_sizes)


def zadanie1d_rbfSVM(class_names, dataloaders, dataset_sizes):

  VGG_16 = create_model_ex1c()
  VGG_16.cuda()

  print("Rbf kernel")
  rbf_svm = load(MODEL_PATH_EX1cRBF)
  eval_svm(VGG_16, rbf_svm, dataloaders, dataset_sizes)

# 1d)
# Zanalizować wyniki klasyfikacji. W szczególności,
# zbadać efekt dopuszczenia błędnych klasyfikacji, porównać z wynikami 1a.
#def zadanie1d():

#############
# ZADANIE 2 #
#############
# 2a)
# Przeprowadzić uczenie ostatniej warstwy splotowej wraz z częścią klasyfikującą
def zadanie2a(class_names, dataloaders, dataset_sizes):

    try:
      os.mkdir(EX2a_FOLDER_PATH)
    except:
      print(EX2a_FOLDER_PATH + ' already exists')

    save_training_params(TRAINING_PARAMS_PATH_EX2a)

    VGG_16 = create_model_ex2a(class_names)

    VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2a, resume_training = RESUME_TRAINING_EX2a)

    VGG_16.cuda()
    summary(VGG_16, SUMMARY_SHAPE)

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, VGG_16.parameters()), lr=LR, momentum = MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma = GAMMA)

    print("Test before training")
    eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
    visualize_model(VGG_16, dataloaders, class_names)
    VGG_16 = train_model(VGG_16, CRITERION, optimizer_ft, exp_lr_scheduler,
     dataloaders, dataset_sizes, HISTORY_PATH_EX2a, RESUME_TRAINING_EX2a, num_epochs=NUMBER_OF_EPOCHS)

    torch.save(VGG_16.state_dict(), MODEL_PATH_EX2a)

    plot_training(HISTORY_PATH_EX2a, PNG_ACCURACY_PATH_EX2a, PNG_LOSS_PATH_EX2a)

# 2b)
# Przeprowadzić uczenie dwóch ostatnich warstw splotowych wraz z częścią 
# klasyfikującą
def zadanie2b(class_names, dataloaders, dataset_sizes):

    try:
      os.mkdir(EX2b_FOLDER_PATH)
    except:
      print(EX2b_FOLDER_PATH + ' already exists')

    save_training_params(TRAINING_PARAMS_PATH_EX2b)

    VGG_16 = create_model_ex2b(class_names)

    VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2b, resume_training = RESUME_TRAINING_EX2b)

    VGG_16.cuda()
    summary(VGG_16, SUMMARY_SHAPE)

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, VGG_16.parameters()), lr=LR, momentum = MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma = GAMMA)

    print("Test before training")
    eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
    visualize_model(VGG_16, dataloaders, class_names)
    VGG_16 = train_model(VGG_16, CRITERION, optimizer_ft, exp_lr_scheduler,
     dataloaders, dataset_sizes, HISTORY_PATH_EX2b, RESUME_TRAINING_EX2b, num_epochs=NUMBER_OF_EPOCHS)

    torch.save(VGG_16.state_dict(), MODEL_PATH_EX2b)

    plot_training(HISTORY_PATH_EX2b, PNG_ACCURACY_PATH_EX2b, PNG_LOSS_PATH_EX2b)

# 2c)
# Wytrenować całą sieć dla zadanych danych
def zadanie2c(class_names, dataloaders, dataset_sizes):

    try:
      os.mkdir(EX2c_FOLDER_PATH)
    except:
      print(EX2c_FOLDER_PATH + ' already exists')

    save_training_params(TRAINING_PARAMS_PATH_EX2c)

    VGG_16 = create_model_ex2c(class_names)

    VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2c, resume_training = RESUME_TRAINING_EX2c)

    VGG_16.cuda()
    summary(VGG_16, SUMMARY_SHAPE)

    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, VGG_16.parameters()), lr=LR, momentum = MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma = GAMMA)

    print("Test before training")
    eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
    visualize_model(VGG_16, dataloaders, class_names)
    VGG_16 = train_model(VGG_16, CRITERION, optimizer_ft, exp_lr_scheduler,
     dataloaders, dataset_sizes, HISTORY_PATH_EX2c, RESUME_TRAINING_EX2c, num_epochs=NUMBER_OF_EPOCHS)

    torch.save(VGG_16.state_dict(), MODEL_PATH_EX2c)

    plot_training(HISTORY_PATH_EX2c, PNG_ACCURACY_PATH_EX2c, PNG_LOSS_PATH_EX2c)

# 2d)
# Uprościć strukturę sieci wytrenowanej w zadaniu 2c (np. poprzez usunięcie 
# jednej lub więcej końcowych warstw splotowych, usunięcie warstw 
# regularyzujących itp.) i ponowić uczenie
def zadanie2d(class_names, dataloaders, dataset_sizes):

    try:
      os.mkdir(EX2d_FOLDER_PATH)
    except:
      print(EX2d_FOLDER_PATH + ' already exists')

    save_training_params(TRAINING_PARAMS_PATH_EX2d)

    VGG_16 = create_model_ex2d(class_names)

    VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2d, resume_training = RESUME_TRAINING_EX2d)

    VGG_16.cuda()
    summary(VGG_16, SUMMARY_SHAPE)
  
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, VGG_16.parameters()), lr=LR, momentum = MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma = GAMMA)

    print("Test before training")
    eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
    visualize_model(VGG_16, dataloaders, class_names)
    VGG_16 = train_model(VGG_16, CRITERION, optimizer_ft, exp_lr_scheduler,
     dataloaders, dataset_sizes, HISTORY_PATH_EX2d, RESUME_TRAINING_EX2d, num_epochs=NUMBER_OF_EPOCHS)

    torch.save(VGG_16.state_dict(), MODEL_PATH_EX2d)

    plot_training(HISTORY_PATH_EX2d, PNG_ACCURACY_PATH_EX2d, PNG_LOSS_PATH_EX2d)

# 2e)
# Zanalizować wyniki 2 abcd
def analiza2a(class_names, dataloaders, dataset_sizes):
  print("Analiza 2a")
  VGG_16 = create_model_ex2a(class_names)

  resume_training = True

  VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2a, resume_training = resume_training)

  print("Test after training")
  pred, labels = eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
  # visualize_model(VGG_16, dataloaders, class_names, num_images = PRESENTATION_BATCH)
  return pred, labels
  
def analiza2b(class_names, dataloaders, dataset_sizes):
  print("Analiza 2b")
  VGG_16 = create_model_ex2b(class_names)

  resume_training = True

  VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2b, resume_training = resume_training)

  print("Test after training")
  eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
  visualize_model(VGG_16, dataloaders, class_names, num_images = PRESENTATION_BATCH)

def analiza2c(class_names, dataloaders, dataset_sizes):
  print("Analiza 2c")
  VGG_16 = create_model_ex2c(class_names)

  resume_training = True

  VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2c, resume_training = resume_training)

  print("Test after training")
  pred, labels = eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
  # visualize_model(VGG_16, dataloaders, class_names, num_images = PRESENTATION_BATCH)
  return pred, labels

def analiza2d(class_names, dataloaders, dataset_sizes):
  print("Analiza 2d")
  VGG_16 = create_model_ex2d(class_names)

  resume_training = True

  VGG_16 = load_weights(VGG_16, MODEL_PATH_EX2d, resume_training = resume_training)

  print("Test after training")
  eval_model(VGG_16, CRITERION, dataloaders, dataset_sizes)
  visualize_model(VGG_16, dataloaders, class_names, num_images = PRESENTATION_BATCH)

def porownanie_modeli_wykres():
  print("Porównanie wykresow")
  models_results = [HISTORY_PATH_EX2a, HISTORY_PATH_EX2b, HISTORY_PATH_EX2c, HISTORY_PATH_EX2d]
  plot_compare_models(models_results, PNG_ACCURACY_PATH_EX2e, PNG_LOSS_PATH_EX2e)

#############
# ZADANIE 3 #
#############
# 3a)
# Dokonać wizualizacji obszarów uwagi sieci wytrenowanych w zadaniu 1 oraz 2
# z wykorzystaniem metod Class Activation Map (CAM)
def zadanie3a():
  print("Zadanie 3a")

# 3b)
# Dokonać wizualizacji aktywacji wewnętrznych warstw sieci z wykorzystaniem
# techniki DeepDream
def zadanie3b():
  print("Zadanie 3b")
