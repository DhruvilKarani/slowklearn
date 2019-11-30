import numpy as np 
import sys
from collections import Counter

def accuracy(y_true, y_pred):
    return np.mean(y_true==y_pred)



def confusion_matrix(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    labels = np.sort(np.unique(y_true))
    num_labels = len(labels)
    cm = np.zeros((num_labels, num_labels))
    for i, true_label in enumerate(labels):
        distribution = y_pred[y_true==true_label]
        for j, pred_label in enumerate(labels):
            cm[i,j] = np.count_nonzero(distribution==pred_label)
    return cm

 

if __name__=='__main__':
    y_true = [0, 1, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 0, 0, 1]
    print(confusion_matrix(y_true, y_pred))    