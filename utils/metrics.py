import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    print(pred.shape)
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

def focus_loss(pred, true, mean):
    pred[pred < mean] = 0
    true[true < mean] = 0
    #print(pred.shape)
    #pred = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2])
    #true = true.reshape(true.shape[0]*true.shape[1]*true.shape[2])
    return pred, true

def range_loss(pred, true):
    err = torch.abs(pred-true)
    
    loss = torch.mul(err, true)
    loss = torch.sum(loss, dim=(0,1,2), keepdim=True)
    loss = loss.squeeze()
    #print(loss)

    return loss
    
    
def range_loss_np(pred, true):
    err = np.abs(pred, true)
    loss = err * true
    loss = np.sum(loss, dim=(0,1,2), keepdim=True)
    print("np:",loss.shape)
    
    return loss
