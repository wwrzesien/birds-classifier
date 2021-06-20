################################ LIBRARIES START ###############################
#External Libraries
import torch
import os
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from importlib import reload

#Internal Libraries
import parameters
parameters = reload(parameters)
from parameters import BATCH_SIZE, NUM_WORKERS, MEAN, STD, TRAIN, TEST, VAL

################################# LIBRARIES END ################################

def create_dataloaders():

  use_gpu = torch.cuda.is_available()
  if use_gpu:
      print("Using CUDA")


  data_dir = '/content/drive/MyDrive/260_Bird_Species'

  batch_size = BATCH_SIZE # The size of input data took for one iteration
  num_workers = NUM_WORKERS # number of subprocess puting data into ram in parallel
  # Normalize image to [0 1] range for each chanell with ImageNet Standard values
  mean = MEAN
  std = STD

  # Image data format for VGG-16 is 224x224 - in our folder 
  # we alredy have images in those sizes and in same fashion
  # so no need to use transforms.RandomResizedCrop(224) or 
  # transforms.Resize(224)
  data_transforms = {
      TRAIN: transforms.Compose([
          # Data augumentation for batches in each iteration / Chain of Image Transformation.
          transforms.RandomHorizontalFlip(), # flip image
          transforms.ColorJitter(), # <- random  # (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1) <- # static
          #transforms.Normalize(mean,std), 
          transforms.ToTensor() # put into tensor  0 < - > 1
      ]),
      VAL: transforms.Compose([
          #transforms.Normalize(mean,std),
          transforms.ToTensor() # put into tensor
      ]),
      TEST: transforms.Compose([
          transforms.RandomHorizontalFlip(), # flip image 
          transforms.ColorJitter(),
          #transforms.Normalize(mean,std), 
          transforms.ToTensor() # put into tensor
      ])
  }

  # Transform and load dataset from train valid and test
  image_datasets = {
      x: datasets.ImageFolder(os.path.join(data_dir, x),transform=data_transforms[x])
      for x in [TRAIN, VAL, TEST]
  }

  # Shuffle and load dataset in parallel into dict batches from train valid and test
  dataloaders = {
      x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=num_workers)
      for x in [TRAIN, VAL, TEST]
  }

  # Sizes of dataset
  dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}


  # print captured data and classes
  for x in [TRAIN, VAL, TEST]:
      print("Loaded {} images under {}".format(dataset_sizes[x], x))

  
  print("Classes: ")
  class_names = image_datasets[TRAIN].classes
  print(image_datasets[TRAIN].classes)
  print('Number of classes: {}'.format(len(class_names)))

  return dataloaders, image_datasets, dataset_sizes


def imshow(inp, title=None):
    # Class image show
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, classes, class_names):
    # show databatch images
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])