import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import warnings
import os
import pickle

l2_loss = nn.MSELoss()

class EE_LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim = 1):
        super(EE_LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size=self.input_dim,
                     hidden_size=self.hidden_dim,
                     num_layers=self.num_layers,
                     batch_first=True,
                     dropout=self.dropout)
        
        self.dense = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        
        out, hidden = self.lstm(x)
        outputs = self.dense(out)
        return outputs

    def test(self,x):
        outputs_list = []
        seq_len = x.size()[1]
        for i in range(seq_len):
            if i == 0:
              xtemp = x[:,i,:].unsqueeze(1)
              xtemp, hidden = self.lstm(xtemp)
            else:
              xtemp = x[:,i,:].unsqueeze(1)
              xtemp, hidden = self.lstm(xtemp, hidden)
            xtemp = self.dense(xtemp)
            outputs_list.append(xtemp)
        outputs = torch.cat(outputs_list,1)
        return outputs

class Energy_Expenditure(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return self.len


def getSourceList(keyword):
    if keyword == 'All':
        source_list = ['ANKLE','THIGH','HR']
    if keyword == 'IMU':
        source_list = ['ANKLE','THIGH']
    if keyword == 'HR':
        source_list = ['HR']

    return source_list

def activity_data_convert(activity_name,data_len):
    if activity_name == 'Walk':
        activity_num = 1
    if activity_name == 'Run':
        activity_num = 2
    if activity_name == 'Stairclimb':
        activity_num = 3
    if activity_name == 'Cycle':
        activity_num = 4

    output_list = activity_num*np.ones(data_len)
    output_list = np.expand_dims(output_list, axis=1)

    return output_list

def loadData(jsi_data, slade_data, source_list, activity_flag):
    data_list = []
    target_list =[]
    activity_list =[]

    # JSI Data
    for filename in os.listdir(jsi_data):
        if 'Person' in filename:
            temp = pd.read_csv(jsi_data+'/'+filename)
            target_cols = [col for col in temp.columns if 'COSMED' in col]
            target_list.append(temp[target_cols].values)
            data_cols = [col for col in temp.columns for j in source_list if j in col ]
            input_data = temp[data_cols].values
            temp_name = filename.split(sep='_')[1]
            activity_name = temp_name.replace(".csv","")
            if activity_flag == True:
                output_list = activity_data_convert(activity_name,len(input_data))
                input_data = np.append(input_data, output_list,1)
            data_list.append(input_data)
            
            activity_list.append(activity_name)

    # Slade Data
    for filename in os.listdir(slade_data):
        if '.csv' in filename:
            temp = pd.read_csv(slade_data+'/'+filename)
            target_cols = [col for col in temp.columns if 'MET' in col and not 'MET HR' in col]
            target_list.append(temp[target_cols].values)
            data_cols = [col for col in temp.columns for j in source_list if j in col ]
            input_data = temp[data_cols].values
            
            #1- quiet standing
            #2- walking at 1.0 m/s
            #3- walking at 1.5 m/s
            #4- running at 2.5 m/s
            #5- running at 3.0 m/s
            #6- climbing stairs at 50 steps/min
            #7- climbing stairs at 70 steps/min
            #8- biking with resistance of 50 Watts
            #9- biking with resistance of 120 Watts 

            if 'C02' in filename or 'C03' in filename:
                activity_name = 'Walk'
            elif 'C04' in filename or 'C05' in filename:
                activity_name = 'Run'
            elif 'C06' in filename or 'C07' in filename:
                activity_name = 'Stairclimb'
            elif 'C08' in filename or 'C09' in filename:
                activity_name = 'Cycle'

            if activity_flag == True:
                output_list = activity_data_convert(activity_name,len(input_data))
                input_data = np.append(input_data, output_list,1)

            activity_list.append(activity_name)
            data_list.append(input_data)

    return data_list, target_list, activity_list

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

    return xx_pad, yy_pad, x_lens, y_lens

def evalModel(keyword, activity_flag, test_loader, device):

    if keyword == 'HR':
        if activity_flag == True:
            features = [0,7]
            model = torch.load('LSTM/HR_Act.pkl')
        else:
            features = [0]
            model = torch.load('LSTM/HR.pkl')
    elif keyword == 'IMU':
        if activity_flag == True:
            features = [1,2,3,4,5,6,7]
            model = torch.load('LSTM/IMU_Act5.pkl')
        else:
            features = [1,2,3,4,5,6]
            model = torch.load('LSTM/IMU.pkl')
    else: # keyword == 'All'
        if activity_flag == True:
            features = [0,1,2,3,4,5,6,7]
            model = EE_LSTM(len(features), 128, 4, 0.1)
            weights = torch.load('LSTM/All_Act.pkl')
            model.load_state_dict(weights)
        else:
            features = [0,1,2,3,4,5,6]
            model = EE_LSTM(len(features), 128, 4, 0.1)
            weights = torch.load('LSTM/All.pkl')
            model.load_state_dict(weights)

    model.to(device)
    model.to(torch.double)

    #################################

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    l1_err = 0.0
    l2_err = 0.0

    model.eval()
    with torch.no_grad():
        for n_batch, [in_batch, label, in_len, label_len] in enumerate(test_loader):
            in_batch, label = in_batch.to(device), label.to(device)
            # Only consider the model features
            in_batch = in_batch[:,:,features]
            # Get prediction
            pred = model.test(in_batch)

            l1_err += l1_loss(pred, label).item()
            l2_err += l2_loss(pred, label).item()

        print('-' * 35)

        if activity_flag == True:
            print(keyword, '+ Activity')
        else:
            print(keyword, 'Only')

        print('MSE Error: %.2f, MAE Error: %.2f' %(l2_err, l1_err))
    
    return pred, label

def initPlot(pred, label, color, leg):
    # Trim the padding
    old_label = label
    label = label[0][old_label[0]!=0]
    pred = pred[0][old_label[0]!=0]

    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    leg = leg = leg + '%.2f' %l2_loss(torch.from_numpy(pred), torch.from_numpy(label)).item()

    ax.plot(range(len(label)), label, color = 'gray', label = 'Ground Truth')
    ax.plot(range(len(label)), pred, '--', color = color, label = leg)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MET Value')

    return fig, ax

def addToPlot(ax, pred, label, color, leg):
    # Trim the padding
    old_label = label
    label = label[0][old_label[0]!=0]
    pred = pred[0][old_label[0]!=0]

    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()

    leg = leg + '%.2f' %l2_loss(torch.from_numpy(pred), torch.from_numpy(label)).item()

    # ax.plot(range(len(label)),label, '-', color = color)
    ax.plot(range(len(label)),pred, '--', color = color, label = leg)
    
    return ax

def main():

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('*' * 50)
    if torch.cuda.is_available():  
        print('CUDA is found! Training on %s...' %torch.cuda.get_device_name(0))
    else:
        warnings.warn('CUDA not found! Training may be slow...')

    jsi_data = '../Models/Data/JSI/1Hz'
    slade_data = '../Models/Data/Slade/1Hz'

    # Call for the full source list and activity_flag = True for now
    source_list = getSourceList('All')
    activity_flag = True
    data_list, target_list, activity_list = loadData(jsi_data, slade_data, source_list, activity_flag)

    test_idx = np.random.choice(len(data_list), len(data_list)//5, replace=False).tolist()
    test_data = [data_list[i] for i in test_idx]
    test_target = [target_list[i] for i in test_idx]

    test = Energy_Expenditure(test_data, test_target)

    test_loader = DataLoader(dataset=test, batch_size=50, shuffle=False, num_workers=2, collate_fn=pad_collate)

    leg_size = 12
    aspect = 'auto'
    # Run all the models
    # HR Only
    pred, label = evalModel('HR', False, test_loader, device)
    fig, ax = initPlot(pred, label, 'green', 'HR, MSE = ')
    pred, label = evalModel('HR', True, test_loader, device)
    ax = addToPlot(ax, pred, label, 'red', 'HR + Act, MSE = ')
    ax.set_title('HR')
    ax.set_aspect(aspect)
    ax.legend(prop = {'size': leg_size})
    fig.savefig('HRPred')
    pickle.dump(fig, open('HR.fig.pkl', 'wb'))
    # IMU
    pred, label = evalModel('IMU', False, test_loader, device)
    fig, ax = initPlot(pred, label, 'orange', 'IMU, MSE = ')
    pred, label = evalModel('IMU', True, test_loader, device)
    ax = addToPlot(ax, pred, label, 'blue', 'IMU + Act, MSE = ')
    ax.set_title('IMU')
    ax.set_aspect(aspect)
    ax.legend(prop = {'size': leg_size})
    fig.savefig('IMUPred')
    pickle.dump(fig, open('IMU.fig.pkl', 'wb'))
    # # ALL
    pred, label = evalModel('All', False, test_loader, device)
    fig, ax = initPlot(pred, label, 'magenta', 'All, MSE = ')
    pred, label = evalModel('All', True, test_loader, device)
    ax = addToPlot(ax, pred, label, 'yellow', 'All + Act, MSE = ')
    ax.set_title('All')
    ax.set_aspect(aspect)
    ax.legend(prop = {'size': leg_size})
    fig.savefig('AllPred')
    pickle.dump(fig, open('All.fig.pkl', 'wb'))

if __name__ == "__main__":
    main()