################################ LIBRARIES START ###############################
# External Libraries
import sklearn
from sklearn import preprocessing, metrics
import time
import copy
import torch
from torch.autograd import Variable
from importlib import reload
import csv
import pickle
import numpy as np

# Internal Libraries
import parameters
parameters = reload(parameters)
from parameters import TRAIN, TEST, VAL, USE_GPU, DEVICE, PRESENTATION_BATCH, \
 TRAIN_SHOW_BATCH_CHANGE, CSV_DELIMITER, CSV_NEWLINE

import data
data = reload(data)
from data import show_databatch

################################# LIBRARIES END ################################

def load_weights(model, model_path, resume_training = False):
  if resume_training:
      print("Loading pretrained model..")
      model.load_state_dict(torch.load(model_path, map_location=DEVICE))

  if USE_GPU:
    model.cuda() # .cuda() will move calculations to the GPU side
  print("Loaded!")
  return model

def train_model(model, criterion, optimizer, scheduler, dataloaders, 
                dataset_sizes, history_path, resume_training, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])
    
    if resume_training:
      MODE = "a" # append to file
    else:
      MODE = "w" # write new file

    f = open(history_path, mode = MODE, newline = CSV_NEWLINE)
    csv_writer = csv.writer(f, delimiter = CSV_DELIMITER)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        
        for i, (inputs, labels) in enumerate(dataloaders[TRAIN]):
            if i % TRAIN_SHOW_BATCH_CHANGE == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break

            if USE_GPU:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            
            # loss function
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train * 2 / dataset_sizes[TRAIN]
        
        model.train(False)
        model.eval()
            
        for i, (inputs, labels) in enumerate(dataloaders[VAL]):
            if i % TRAIN_SHOW_BATCH_CHANGE == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
            
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
              outputs = model(inputs)
            
              _, preds = torch.max(outputs.data, 1)
              loss = criterion(outputs, labels)
            
              loss_val += loss.item()
              acc_val += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / dataset_sizes[VAL]
        avg_acc_val = acc_val / dataset_sizes[VAL]
        
        print()
        print("Epoch {} result: ".format(epoch+1))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        csv_writer.writerow([epoch+1, avg_acc.item(), avg_loss, avg_acc_val.item(), avg_loss_val])
    f.close()
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders, class_names, num_images=6):
    training_state = model.training
    
    # Set model for evaluation
    model.train(False)
    model.eval()

    (inputs, labels) = next(iter(dataloaders[TEST]))

    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
            
        _, preds = torch.max(outputs.data, 1)

    pres_imgs = inputs[0:PRESENTATION_BATCH].data.cpu()
    pres_real_labels = labels[0:PRESENTATION_BATCH]
    pres_pred_labels = preds[0:PRESENTATION_BATCH]

    print("Ground truth:")
    show_databatch(pres_imgs, pres_real_labels, class_names)
    print("Prediction:")
    show_databatch(pres_imgs, pres_pred_labels, class_names)    
    model.train(mode = training_state) # Revert model back to original training state


def eval_model(model, criterion, dataloaders, dataset_sizes):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    predicts = []
    lbls = []
    
    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)
    
    model.train(False)
    model.eval()

    aaa = True

    for i, (inputs, labels) in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_test += loss.item()
            acc_test += torch.sum(preds == labels.data)

            if aaa:
              print('\nAAAAA')
              print(preds)
              print(labels.data)
              aaa = False
              print(torch.sum(preds == labels.data))
            predicts = np.concatenate((predicts, preds.cpu()), axis=None)
            lbls = np.concatenate((lbls, labels.data.cpu()), axis=None)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    print("acc test", acc_test)
    print("dataset size", dataset_sizes[TEST])
    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test / dataset_sizes[TEST]
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)
    return predicts, lbls


def save_checkpoint(model, version):
  """Save model data"""
  checkpoint = {
      "model_state_dict": model.state_dict() # weights from layers
  }
  filename = 'model-{}-{}.pth'.format(version, time.strftime("%Y%m%d-%H%M"))
  torch.save(checkpoint, filename)


def train_svm(model, svm, dataloaders):

  train_batches = len(dataloaders[TRAIN])
  labels_list=[]
  outputs_list = []

  model.train(False)
  model.eval()

  # EVALUATE THGROUGH MODEL
  with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders[TRAIN]):
      if i % TRAIN_SHOW_BATCH_CHANGE == 0:
        print("\rEvaluating batch {}/{}".format(i, train_batches * 1/2), end='', flush=True)
      # Use half training dataset
      if i >= train_batches * 1/2:
        break
      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

      output = model(inputs)

      output = output.cpu()
      labels = labels.cpu()

      outputs_list.append(output)
      labels_list.append(labels)

  print('\nCreating outputs')
  outputs_svm = outputs_list[0]
  labels_list_svm = labels_list[0]

  outputs_list = outputs_list[1:]
  labels_list = labels_list[1:]

  for item in outputs_list:
    outputs_svm = torch.cat((outputs_svm, item),0)

  for item in labels_list:
    labels_list_svm = torch.cat((labels_list_svm, item),0)


  print("\noutputs shape: ", outputs_svm.shape)
  print("labels list shape: ", labels_list_svm.shape)

  # Save svm input || Don't - eats so much disc space
  # data = {"outputs_svm": outputs_svm, "labels_list_svm":labels_list_svm}
  # with open('/content/drive/MyDrive/260_Bird_Species/Model_weights/input_svm2-5.pickle', 'wb') as f:
  #   pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

  print('Fitting SVM')
  svm.fit(outputs_svm, labels_list_svm)
  print('SVM learned!')
  return svm

def eval_svm(model, svm, dataloaders, dataset_sizes):
  since = time.time()
  avg_loss = 0
  avg_acc = 0
  loss_test = 0
  acc_test = 0
  
  test_batches = len(dataloaders[TEST])
  print("Evaluating model")
  print('-' * 10)
  
  model.train(False)
  model.eval()

  outputs_list = []
  labels_list = []
  with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders[TEST]):
      if i % TRAIN_SHOW_BATCH_CHANGE == 0:
        print("\rEvaluating batch {}/{}".format(i, test_batches), end='', flush=True)

      inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

      output = model(inputs)

      output = output.cpu()
      labels = labels.cpu()

      outputs_list.append(output)
      labels_list.append(labels)

  print('\nCreating outputs')
  outputs_svm = outputs_list[0]
  labels_list_svm = labels_list[0]

  outputs_list = outputs_list[1:]
  labels_list = labels_list[1:]

  for item in outputs_list:
    outputs_svm = torch.cat((outputs_svm, item),0)

  for item in labels_list:
    labels_list_svm = torch.cat((labels_list_svm, item),0)

  # Save svm input || Don't - eats so much disc space
  # data = {"outputs_svm": outputs_svm, "labels_list_svm":labels_list_svm}
  # with open('/content/drive/MyDrive/260_Bird_Species/Model_weights/input_svm.pickle', 'wb') as f:
  #   pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

  preds = svm.predict(outputs_svm)
  acc_test = metrics.accuracy_score(preds, labels_list_svm)
  
  elapsed_time = time.time() - since
  print()
  print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
  print("Avg acc (test): {:.4f}".format(acc_test))
  print('-' * 10)
