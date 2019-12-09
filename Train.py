# Ver.1.0.1
# added roc curve, 
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import time
import torch
import torchvision
import torch.nn.functional as F
import torchvision.models as models
import sys
import scikitplot as skplt

from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.optim import lr_scheduler
from scipy import interp
from itertools import cycle

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#cc mean/std
#train: tensor([0.1047, 0.1047, 0.1047]) tensor([0.1516, 0.1516, 0.1516])
#val : tensor([0.1077, 0.1077, 0.1077]) tensor([0.1533, 0.1533, 0.1533])

#mlo mean/std
#train: tensor([0.1128, 0.1128, 0.1128]) tensor([0.1744, 0.1744, 0.1744]) 
#val: tensor([0.1102, 0.1102, 0.1102]) tensor([0.1776, 0.1776, 0.1776])


def calculate_(dataloaders):
    
    dataloader = dataloaders
    mean = 0.
    std = 0.
    
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    
    
    return mean, std

    
def load_data(train_mean,train_std, val_mean, val_std):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
            ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(val_mean, val_std)
            ])
        }
    
    return data_transforms

def load_model():
    
    model = models.resnet18(pretrained = True)
    
    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Classifier architecture to put on top of resnet18
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(173056, 100)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.3)),
        ('fc2', nn.Linear(100, 4)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    model.fc = fc
    model.to(device)
    
    return model

def set_optimizer(model, opt,lr):
    
    if opt == 'sgd':
        optimizer = optim.SGD(model.fc.parameters(), lr = lr, momentum=0.9)
    elif opt== 'adam':
        optimizer = optim.Adam(model.fc.parameters(), lr = lr, weight_decay=0.000006)
    else:
        optimizer = optim.Adagrad(model.fc.parameters(), lr = lr, weight_decay =0.000005 )
    
    return optimizer 

def plot_graph(train_losses, test_losses, train_accs, test_accs, new_name):
    
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_losses, label= 'Training_loss')
    plt.plot(train_accs, label='Training acc')
    plt.plot(test_losses, label= 'Validation loss')
    plt.plot(test_accs, label='Validation acc')
    
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(frameon=False)
    
    # save the figure
    plt.savefig(os.path.join(os.getcwd(), "models", new_name, new_name + '_Acc.png'))
    plt.close

def plot_PR_curve(y_score,y_test,new_name):
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = 4
    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = y_score.shape
    X = np.c_[y_score, random_state.randn(n_samples, 200 * n_features)]

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")
    
    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],label='PR curve of class {0} (area = {1:0.2f})'
                       ''.format(i+1, average_precision[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('RECALL')
    plt.ylabel('PRECISION')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="upper right")
#     plt.show()
    # save the figure
    plt.savefig(os.path.join(os.getcwd(), "models", new_name, new_name + 'PR_curve.png'))
    plt.close


def pot_ROC_curve(y_test, y_score,new_name):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 4
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    lw =2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate( specificity)')
    plt.ylabel('True Positive Rate(Sensitive)')
    plt.title('ROC to multi-class')
    plt.legend(loc="lower right")
    # save the figure
    plt.savefig(os.path.join(os.getcwd(), "models", new_name, new_name + 'ROC_curve.png'))
    plt.close

#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(optimizer, epoch):

    lr = 0.001

    if epoch > 160:
        lr = lr / 10000
    elif epoch > 120:
        lr = lr / 1000
    elif epoch > 80:
        lr = lr / 100
    elif epoch > 40:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test(model,dataloaders,criterion,dataset_sizes):
    
    model.eval()
    test_acc = 0.0
    test_loss = 0.0 
    
    y_true  = []
    y_pred  = []
    y_score = []     # Probabilities
        
    for i, (inputs_, labels_) in enumerate(dataloaders['val']):
        inputs_ = inputs_.to(device)
        labels_ = labels_.to(device)


        # Predict classes using images from the test dataset
        outputs = model(inputs_)        
        test_probs, test_prediction = torch.max(outputs, 1)
        loss = criterion(outputs, labels_)
        
        
        # calcuate test acc, loss
        test_acc += torch.sum(test_prediction == labels_.data)
        test_loss += loss.item()*inputs_.size(0)
        
        test_prediction = test_prediction.cpu().numpy()
        labels_ = labels_.cpu().numpy()

        with torch.no_grad():
            y_pred.extend(test_prediction)
            y_true.extend(labels_)

            sm =  torch.nn.Softmax()
            prob_arr = sm(outputs)
            prob_arr = prob_arr.cpu().numpy()

            # Save probabilities in 'y_score' array
            y_score.extend(prob_arr)
    
    #Compute the average acc and loss over all images
    test_acc = test_acc.double() / dataset_sizes['val']
    test_loss = test_loss / dataset_sizes['val']
    
    # Print confusion matrix, clf_report
    cf_matrix = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred)
    
    print ('========================')
    print ("Confusion Mmatrix")
    print (cf_matrix)
    print (clf_report)
    
    return test_acc, test_loss, y_score, y_pred

def train_model(model,new_name, num_epochs, opt, criterion, dataloaders, dataset_sizes):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model_path = os.path.join(os.getcwd(), 'models', new_name)
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []


    # Decay LR by a factor of 0.01 every 50 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(opt, step_size=50, gamma=0.01)

    for epoch in range(num_epochs):
        logs = {}
        
        print ('========================')
        print ("Epoch {} / {}".format(epoch, num_epochs -1))
        
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # clear all accumulated gradients
            opt.zero_grad()
            
            #Predict classes using images from the test set
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Adjust parameters according to the computed grdients
            opt.step()
#             exp_lr_scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            
            probs, prediction = torch.max(outputs, 1)
    
            train_acc += torch.sum(prediction == labels.data)
            
        # Call the learning rate adjustment function
#         adjust_learning_rate(opt, epoch)
        train_acc  = train_acc.double() / dataset_sizes['train']
        train_loss = train_loss / dataset_sizes['train'] 
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on the test set
        test_acc, test_loss, y_score, y_pred = test(model,dataloaders,criterion,dataset_sizes)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
        # Save the model if the best acc is greater than our current best
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
#             torch.save(model.state_dict(best_model_wts),os.path.join(model_path, "Resnet18_model_{}.pth".format(epoch)))    
            torch.save(model.state_dict(), os.path.join(model_path, "Resnet18_model_{}.pth".format(epoch)))        
        # Print the metrics
        print ('Train Acc: {:.4f} Train_loss: {:.4f}'.format(train_acc,train_loss))
        print ('Test Acc: {:.4f} Test_loss: {:.4f}'.format(test_acc,test_loss))
    
    time_elapsed = time.time() - since
    print ('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed %60)) 


    
    # Binarize the output
    y_test = label_binarize(y_pred, classes=[1,2,3,4])
    y_score = np.array(y_score)
    
    ### Plot Precision-Recall and plot curve
    plot_PR_curve(y_score,y_test,new_name)
    pot_ROC_curve(y_test, y_score,new_name)
    plot_graph(train_losses, test_losses, train_accs, test_accs, new_name)
    torch.save(model.state_dict(best_model_wts), os.path.join(model_path, new_name +'.pth'))

    return model
    

def train(args):
    
    now = datetime.now()
    dt_string = now.strftime("%m%d")
    
    data_dir = args.data_dir      # 
    batch_size = int(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    num_epochs = int(args.num_epochs)
    optimizer = args.optimizer
    lr = float(args.lr)
    
    new_name = str(dt_string) +"_" + args.model_name + "_" + str(batch_size) + "_" + str(num_epochs) + "_" + str(optimizer)+"_" + str(lr)
    
    if (not os.path.exists(os.path.join(os.getcwd(), "models", new_name))):
        os.mkdir(os.path.join(os.getcwd(), "models", new_name))

    data_transforms = {'train': transforms.Compose([transforms.ToTensor()]),
                        'val': transforms.Compose([transforms.ToTensor()])}
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train', 'val']}
    
    dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=4)
                  for x in ['train', 'val']}


    ## Calculate mean , std value 
    train_mean, train_std = calculate_(dataloaders['train'])
    val_mean, val_std = calculate_(dataloaders['val'])
    
    dt = load_data(train_mean,train_std, val_mean, val_std)
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x), dt[x]) for x in ['train', 'val']}
    
    dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size, shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    

    model = load_model()
    opt = set_optimizer(model, optimizer, lr)
    print (opt)
    print (model)

    train_model(model,new_name, num_epochs, opt, criterion, dataloaders, dataset_sizes)
    
    
if __name__ == '__main__':
    
    a = argparse.ArgumentParser()
    a.add_argument("--data_dir")
    a.add_argument("--batch_size", default=64)
    a.add_argument("--num_epochs", default=5)
    a.add_argument("--optimizer", default ='sgd')
    a.add_argument("--model_name")
    a.add_argument("--lr", default=0.001)
    args = a.parse_args()
    
    if (not os.path.exists(args.data_dir)):
        print ("directories do not exist")
        sys.exit(1)
        
    train(args)

