################################ LIBRARIES START ###############################
# External Libraries
from importlib import reload
import csv
import matplotlib.pyplot as plt
# Internal Libraries
import parameters
parameters = reload(parameters)
from parameters import CSV_DELIMITER, CSV_NEWLINE, TRAIN, VAL, NUMBER_OF_EPOCHS, \
  BATCH_SIZE, LR, MOMENTUM, GAMMA

################################# LIBRARIES END ################################

def plot_training(history_path, accuracy_png_path, loss_png_path):
  f = open(history_path, newline=CSV_NEWLINE, mode="r")
  csv_reader = csv.reader(f, delimiter = CSV_DELIMITER)

  epochs = []
  avg_acc_train = []
  avg_loss_train = []
  avg_acc_val =[]
  avg_loss_val = []

  for row in csv_reader:
    epochs.append(row[0])
    avg_acc_train.append(row[1])
    avg_loss_train.append(row[2])
    avg_acc_val.append(row[3])
    avg_loss_val.append(row[4])
  
  epochs = [int(x) for x in epochs]
  avg_acc_train = [float(x) for x in avg_acc_train]
  avg_acc_val = [float(x) for x in avg_acc_val]
  avg_loss_train = [float(x) for x in avg_loss_train]
  avg_loss_val = [float(x) for x in avg_loss_val]


  plt.plot(epochs, avg_acc_train, 'go', epochs, avg_acc_val, 'ro')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend([TRAIN, VAL], loc='upper left')
  plt.savefig(accuracy_png_path)

  plt.clf()
  plt.plot(epochs, avg_loss_train, 'go', epochs, avg_loss_val, 'ro')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend([TRAIN, VAL], loc='upper left')
  plt.savefig(loss_png_path)

def plot_compare_models(paths, accuracy_png_path, loss_png_path):
  data = []

  for path in paths:
    f = open(path, newline=CSV_NEWLINE, mode="r")
    csv_reader = csv.reader(f, delimiter = CSV_DELIMITER)

    d = {
      "epochs": [],
      "avg_acc_train": [],
      "avg_loss_train": [],
      "avg_acc_val": [],
      "avg_loss_val": []
    }

    for row in csv_reader:
      d["epochs"].append(row[0])
      d["avg_acc_train"].append(row[1])
      d["avg_loss_train"].append(row[2])
      d["avg_acc_val"].append(row[3])
      d["avg_loss_val"].append(row[4])
    
    d["epochs"] = [int(x) for x in d["epochs"]]
    d["avg_acc_train"] = [float(x) for x in d["avg_acc_train"]]
    d["avg_acc_val"] = [float(x) for x in d["avg_acc_val"]]
    d["avg_loss_train"] = [float(x) for x in d["avg_loss_train"]]
    d["avg_loss_val"] = [float(x) for x in d["avg_loss_val"]]

    data.append(d)

  legend = ["ostatnia warstwa", "2 ostatnie warstwy", "cała sieć", "uproszczona struktura"]
  
  fig, (acc, loss) = plt.subplots(1, 2, figsize=(20, 8))
  fig.suptitle('Zbiór treningowy',size =  24)
  acc.plot(data[0]["epochs"], data[0]["avg_acc_train"], '-b')
  acc.plot(data[0]["epochs"], data[1]["avg_acc_train"], '-g')
  acc.plot(data[0]["epochs"], data[2]["avg_acc_train"], '-r')
  acc.plot(data[0]["epochs"], data[3]["avg_acc_train"], '-c')
  loss.plot(data[0]["epochs"], data[0]["avg_loss_train"], '-b')
  loss.plot(data[0]["epochs"], data[1]["avg_loss_train"], '-g')
  loss.plot(data[0]["epochs"], data[2]["avg_loss_train"], '-r')
  loss.plot(data[0]["epochs"], data[3]["avg_loss_train"], '-c')

  # acc.set(xlabel='Epoch', ylabel='Accuracy')
  # loss.set(xlabel='Epoch', ylabel='Loss')
  acc.set_xlabel('Epoch',fontsize = 20)
  acc.set_ylabel('Accuracy',fontsize = 20)
  loss.set_xlabel('Epoch',fontsize = 20)
  loss.set_ylabel('Loss',fontsize = 20)

  acc.legend(legend, loc='lower right',fontsize = 18)
  loss.legend(legend, loc='upper right',fontsize = 18)
  
  fig.savefig(accuracy_png_path + "_train_combine.png")

  # plt.clf()
  # fig, (acc, loss) = plt.subplots(1, 2, figsize=(20, 8))
  # fig.suptitle('Zbiór walidacyjny',size = 24)
  # acc.plot(data[0]["epochs"], data[0]["avg_acc_val"], '-b')
  # acc.plot(data[0]["epochs"], data[1]["avg_acc_val"], '-g')
  # acc.plot(data[0]["epochs"], data[2]["avg_acc_val"], '-r')
  # acc.plot(data[0]["epochs"], data[3]["avg_acc_val"], '-c')
  # loss.plot(data[0]["epochs"], data[0]["avg_loss_val"], '-b')
  # loss.plot(data[0]["epochs"], data[1]["avg_loss_val"], '-g')
  # loss.plot(data[0]["epochs"], data[2]["avg_loss_val"], '-r')
  # loss.plot(data[0]["epochs"], data[3]["avg_loss_val"], '-c')

  # # acc.set(xlabel='Epoch', ylabel='Accuracy')
  # # loss.set(xlabel='Epoch', ylabel='Loss')
  # acc.set_xlabel('Epoch',fontsize = 20)
  # acc.set_ylabel('Accuracy',fontsize = 20)
  # loss.set_xlabel('Epoch',fontsize = 20)
  # loss.set_ylabel('Loss',fontsize = 20)

  acc.legend(legend, loc='lower right',fontsize = 18)
  loss.legend(legend, loc='upper right',fontsize = 18)
  
  fig.savefig(accuracy_png_path + "_val_combine.png")
  
  # plt.plot(data[0]["epochs"], data[0]["avg_acc_train"], '-b')
  # plt.plot(data[0]["epochs"], data[1]["avg_acc_train"], '-g')
  # plt.plot(data[0]["epochs"], data[2]["avg_acc_train"], '-r')
  # plt.plot(data[0]["epochs"], data[3]["avg_acc_train"], '-c')
  # plt.ylabel('Accuracy')
  # plt.xlabel('Epoch')
  # plt.legend(legend, loc='lower right')
  # plt.savefig(accuracy_png_path + "_train.png")

  # plt.clf()
  # plt.plot(data[0]["epochs"], data[0]["avg_acc_val"], '-b')
  # plt.plot(data[0]["epochs"], data[1]["avg_acc_val"], '-g')
  # plt.plot(data[0]["epochs"], data[2]["avg_acc_val"], '-r')
  # plt.plot(data[0]["epochs"], data[3]["avg_acc_val"], '-c')
  # plt.ylabel('Accuracy')
  # plt.xlabel('Epoch')
  # plt.legend(legend, loc='lower right')
  # plt.savefig(accuracy_png_path + "_val.png")

  # plt.clf()
  # plt.plot(data[0]["epochs"], data[0]["avg_loss_train"], '-b')
  # plt.plot(data[0]["epochs"], data[1]["avg_loss_train"], '-g')
  # plt.plot(data[0]["epochs"], data[2]["avg_loss_train"], '-r')
  # plt.plot(data[0]["epochs"], data[3]["avg_loss_train"], '-c')
  # plt.ylabel('Loss')
  # plt.xlabel('Epoch')
  # plt.legend(legend, loc='upper right')
  # plt.savefig(loss_png_path + "_train.png")

  # plt.clf()
  # plt.plot(data[0]["epochs"], data[0]["avg_loss_val"], '-b')
  # plt.plot(data[0]["epochs"], data[1]["avg_loss_val"], '-g')
  # plt.plot(data[0]["epochs"], data[2]["avg_loss_val"], '-r')
  # plt.plot(data[0]["epochs"], data[3]["avg_loss_val"], '-c')
  # plt.ylabel('Loss')
  # plt.xlabel('Epoch')
  # plt.legend(legend, loc='upper right')
  # plt.savefig(loss_png_path + "_val.png")
  

def save_training_params(params_path):
  f = open(params_path, mode = "w")
  f.write("NUMBER_OF_EPOCHS = {}\n".format(NUMBER_OF_EPOCHS))
  f.write("BATCH_SIZE = {}\n".format(BATCH_SIZE))
  f.write("LR = {}\n".format(LR))
  f.write("MOMENTUM = {}\n".format(MOMENTUM))
  f.write("GAMMA = {}".format(GAMMA))
  f.close()
