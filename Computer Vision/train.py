from __future__ import print_function
from __future__ import division

import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from utils import set_parameter_requires_grad, plot_classification_report

from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import logging
import copy
import time
from config import device
from data import get_data_loaders


from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import aim
import argparse
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logger.info("PyTorch Version: ",torch.__version__)
# logger.info("Torchvision Version: ",torchvision.__version__)

def train_model(model: nn.Module, dataloaders:dict, criterion:torch.nn, optimizer: torch.optim.Optimizer,\
     num_epochs=25, is_inception=False, aim_run: aim.Run = None):
    
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    labels_all = []
    predictions_all = []

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)

                        loss = loss1 + 0.4*loss2

                        if aim_run is not None:
                            aim_run.track(loss1,name=f"Cross Entropy",context={"type": f"Batch loss 1"},epoch=epoch)
                            aim_run.track(loss1,name=f"Cross Entropy",context={"type": f"Batch loss 2"},epoch=epoch)
                            aim_run.track(loss,name=f"Cross Entropy",context={"type": f" Batch total"},epoch=epoch)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        if aim_run is not None:
                            aim_run.track(loss,name=f"Cross Entropy",context={"type": f" Batch total"},epoch=epoch)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if aim_run is not None:

                aim_run.track(epoch_loss,name=f"Cross Entropy",context={"type": f" Epoch total"},epoch=epoch)
                aim_run.track(epoch_acc,name=f"Accuracy",context={"type": f" Epoch Accuracy"},epoch=epoch)


            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if aim_run is not None:
                    aim_run['best_acc'] = best_acc.item()

            if phase == 'test':
                
                predictions = model(inputs)
                _, predictions = torch.max(predictions, 1)
                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()

                labels_all += labels.tolist()
                predictions_all += predictions.tolist()

                val_acc_history.append(epoch_acc)


    report = classification_report(labels_all, predictions_all)  
    cf_matrix = confusion_matrix(labels_all, predictions_all)
    print(report)

    ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ## Display the visualization of the Confusion Matrix.
    # plt.show()

    fig = ax.get_figure()
    fig.savefig('confusion_matrix.png')
    aim_run.track(aim.Figure(fig), name = 'Metric',  context = {'type': 'Confusion Matrix'})

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, report


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        logger.info("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--model_name', type=str, default='squeezenet', help='model name')
    parser.add_argument('--data_dir', type=str, default='Images', help='data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--feature_extract', type=int, default=1, help='Flags for feature extracting')

    args = parser.parse_args()

    model_ft, input_size = initialize_model(args.model_name, args.num_classes, args.feature_extract, use_pretrained=True)
    image_datasets, dataloaders_dict = get_data_loaders(args.data_dir, input_size, args.batch_size)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    logger.info("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                logger.info(f"\t ,{name}")
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                logger.info(f"\t ,{name}")


    aim_run = aim.Run(repo='.', experiment=f'{args.model_name}', run_hash=None)
    aim_run['config'] = vars(args)


    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist, fig = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, \
        num_epochs=args.num_epochs, is_inception=(args.model_name=="inception"), aim_run=aim_run)




