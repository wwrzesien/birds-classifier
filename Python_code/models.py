################################ LIBRARIES START ###############################
#External Libraries
import torchvision.models as models
import torch
import torch.nn as nn
from sklearn.svm import SVC
#Internal Libraries

################################# LIBRARIES END ################################
class Identity(torch.nn.Module):
  def __init__(self):
      super(Identity, self).__init__()

  def forward(self, x):
      return x

# MODEL 1a
def create_model_ex1a(class_names):
    # VGG-16 Model for finetuning and feature extraction
    VGG_16 = models.vgg16(pretrained=True)

    # Freeze training for all layers
    for param in VGG_16.features.parameters():
      param.requires_grad = False

    # changing last layer
    # newly created modules have require_grad=True by default
    # in pytorch the vgg network is divided into features (30 deep layers) and classifier. We disabled above only training features. In classifier we have to change last layer to provide as output 260 classes not default 1000. 
    num_features = VGG_16.classifier[6].in_features
    features = list(VGG_16.classifier.children())[:-1] # Remove last layer
    
    # #Unfreeze whole classifier section
    # for child in VGG_16.classifier.children():
    #   for param in child.parameters():
    #     param.requires_grad = True

    # adding fully connected layer
    features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with features outputs - in this case 260 Birds Species (this must be changed as we have 260 classes, not 1000 as default)
    VGG_16.classifier = nn.Sequential(*features) # Replace the model classifier features
    
    return VGG_16

# MODEL 1c
def create_model_ex1c():
  # VGG-16 Model for finetuning and feature extraction
  VGG_16 = models.vgg16(pretrained=True)

  # Freeze training for all layers
  for param in VGG_16.parameters():
    param.requires_grad = False
  #features = list(VGG_16.children())[:-1] # deleting whole classifier section
  #features.extend([nn.Flatten()])
  #VGG_16 = nn.Sequential(*features)
  #VGG_16.classifier = nn.Sequential(*list(VGG_16.classifier.children())[:-1])
  VGG_16.classifier._modules['6'] = nn.Flatten()

  return VGG_16

# MODEL 2a
def create_model_ex2a(class_names):
  # VGG-16 Model for finetuning and feature extraction
  VGG_16 = models.vgg16(pretrained=True)

  # Freeze training for all layers except last one
  for param in VGG_16.features.parameters():
    param.requires_grad = False

  #Unfreeze last layer Conv2d
  i = 0
  for child in VGG_16.features.children():
    i += 1
    if i == 29:
      for param in child.parameters():
        param.requires_grad = True
    
  #print("VGG_16 features children: ", i)

  # changing last layer
  # newly created modules have require_grad=True by default
  # in pytorch the vgg network is divided into features (30 deep layers) and classifier. We disabled above only training features. In classifier we have to change last layer to provide as output 260 classes not default 1000. 
  num_features = VGG_16.classifier[6].in_features
  features = list(VGG_16.classifier.children())[:-1] # Remove last layer

  # adding fully connected layer
  features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with features outputs - in this case 260 Birds Species (this must be changed as we have 260 classes, not 1000 as default)
  VGG_16.classifier = nn.Sequential(*features) # Replace the model classifier features

  return VGG_16


# MODEL 2b
def create_model_ex2b(class_names):
  # VGG-16 Model for finetuning and feature extraction
  VGG_16 = models.vgg16(pretrained=True)

  # Freeze training for all layers except last one
  for param in VGG_16.features.parameters():
    param.requires_grad = False

  #Unfreeze 2 last Conv2d layers
  i = 0
  for child in VGG_16.features.children():
    i += 1
    if i == 27 or i == 29:
      for param in child.parameters():
        param.requires_grad = True
    
  #print("VGG_16 features children: ", i)

  # changing last layer
  # newly created modules have require_grad=True by default
  # in pytorch the vgg network is divided into features (30 deep layers) and classifier. We disabled above only training features. In classifier we have to change last layer to provide as output 260 classes not default 1000. 
  num_features = VGG_16.classifier[6].in_features
  features = list(VGG_16.classifier.children())[:-1] # Remove last layer

  # adding fully connected layer
  features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with features outputs - in this case 260 Birds Species (this must be changed as we have 260 classes, not 1000 as default)
  VGG_16.classifier = nn.Sequential(*features) # Replace the model classifier features

  return VGG_16


# MODEL 2c
def create_model_ex2c(class_names):
  # VGG-16 Model for finetuning and feature extraction
  VGG_16 = models.vgg16(pretrained=True)

  # Don't Freeze 

  # changing last layer
  # newly created modules have require_grad=True by default
  # in pytorch the vgg network is divided into features (30 deep layers) and classifier. We disabled above only training features. In classifier we have to change last layer to provide as output 260 classes not default 1000. 
  num_features = VGG_16.classifier[6].in_features
  features = list(VGG_16.classifier.children())[:-1] # Remove last layer

  # adding fully connected layer
  features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with features outputs - in this case 260 Birds Species (this must be changed as we have 260 classes, not 1000 as default)
  VGG_16.classifier = nn.Sequential(*features) # Replace the model classifier features

  return VGG_16


# MODEL 2d
def create_model_ex2d(class_names):



#  class Upsample(nn.Module):
#    def __init__(self,  scale_factor):
#        super(Upsample, self).__init__()
#        self.scale_factor = scale_factor
#    def forward(self, x):
#        return F.interpolate(x, scale_factor=self.scale_factor)
  # VGG-16 Model for finetuning and feature extraction
  VGG_16 = models.vgg16(pretrained=True)

  # Remove sectors from features
  #features_no_last_conv = list(VGG_16.features.children())[:28]
  VGG_16.features._modules['21'] = Identity()
  VGG_16.features._modules['22'] = Identity()
  VGG_16.features._modules['24'] = Identity()
  VGG_16.features._modules['25'] = Identity()
  VGG_16.features._modules['26'] = Identity()
  VGG_16.features._modules['27'] = Identity()
  VGG_16.features._modules['30'] = Identity()
  VGG_16.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
  #VGG_16.classifier._modules['0'] = nn.AdaptiveAvgPool2d(output_size=(3, 3))
  #VGG_16.classifier._modules['0'] = nn.functional.interpolate(VGG_16.avgpool, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
  VGG_16.classifier._modules['0'] = nn.Linear(in_features=4608, out_features=4608, bias=True)
  VGG_16.classifier._modules['1'] = nn.ReLU(inplace=True)
  VGG_16.classifier._modules['2'] = nn.Dropout(p=0.5,inplace=False)
  VGG_16.classifier._modules['3'] = nn.Linear(in_features=4608, out_features=4096, bias=True)
  VGG_16.classifier._modules['4'] = nn.ReLU(inplace=True)
  VGG_16.classifier._modules['5'] = nn.Dropout(p=0.5,inplace=False)
  VGG_16.classifier._modules['6'] = nn.Linear(in_features=4096, out_features=260, bias=True)


  # Add MaxPool2d
  #features_no_last_conv.extend([nn.MaxPool2d(kernel_size = 2, stride = 2,
  # padding = 0, dilation = 1, ceil_mode = False)])

  
  #features_class = list(VGG_16.classifier.children())[:-1] # remove last
  #in_last_class = VGG_16.classifier[6].in_features
  #features_class.extend([nn.Linear(in_last_class, len(class_names))])

  #VGG_16.features = nn.Sequential(*features_no_last_conv) # Replace the model features features
  #VGG_16.classifier = nn.Sequential(*features_class) # Replace the model classifier features
  print(VGG_16)
  return VGG_16
