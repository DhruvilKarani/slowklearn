import numpy as np 

def mse(y_true, y_pred):
    '''
    --Computes mean squared error between two numpy arrays

    --params:
        y_true: ground truth labels
        y_pred: predicted labels

    --returns:
        mse
    '''
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    assert y_true.shape == y_pred.shape, "Shape of predicted and true labels is unequal"

    return np.mean(np.power(y_true-y_pred, 2))

def cat_crossentropy(y_true, y_pred):
    '''
        Computes categorical cross entropy loss between a one hot encoded label
        and a softmax output

        --params:
            y_true > one-hot encoding
            y_pred > softmax probs
        
        --output
            loss (float)
    '''
    y_pred[y_pred==0] += 10e-4
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()
    log_prob = -np.log(y_pred)
    return np.matmul(y_true, log_prob.T)



if __name__ == '__main__':
    y = np.array([0.1, 0.5, 0.4])
    t = np.array([1, 0, 0])
    print(cat_crossentropy(t, y))
