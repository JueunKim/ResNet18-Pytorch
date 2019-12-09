import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc,accuracy_score,f1_score, recall_score, precision_score

device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


cur_path = os.getcwd()
batch_size = 64
data_dir = '/home//work/data/val'
model_path = os.path.join(cur_path, 'models', 'test.pth')
print (model_path)


def load_model(model_path):
    
    model = models.resnet18(pretrained = True)
    
    for param in model.parameters():
            param.requires_grad = False
    
    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(173056, 100)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.3)),
        ('fc2', nn.Linear(100,4)),
        ('output', nn.LogSoftmax(dim=1))
    ]))  
    model.fc = fc
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    return model 


if __name__ == "__main__":
    model = load_model(model_path)
    model.eval()
    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    ])

    test_data = datasets.ImageFolder(data_dir, transform = test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)
    
    test_acc = 0.0
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_score = []
    
    criterion = nn.CrossEntropyLoss()
    
    for i, (inputs_, labels_) in enumerate(test_loader):
        inputs_ = inputs_.to(device)
        labels_ = labels_.to(device)
        
        outputs = model(inputs_)
        test_probs, test_prediction = torch.max(outputs, 1)
        loss = criterion(outputs, labels_)
        
        
        test_prediction = test_prediction.cpu().numpy()
        labels_ = labels_.cpu().numpy()
        
        with torch.no_grad():
            y_pred.extend(test_prediction)
            y_true.extend(labels_)
            
            sm =  torch.nn.Softmax()
            prob_arr = sm(outputs)
            prob_arr = prob_arr.cpu().numpy()
            
            y_score.extend(prob_arr)
            
    cf_matrix = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred)
    
    print (cf_matrix)
    print (clf_report)
    print ('F1 score', f1_score(y_true, y_pred, average= "micro"))
    print ('Recall', recall_score(y_true, y_pred, average='micro'))
        