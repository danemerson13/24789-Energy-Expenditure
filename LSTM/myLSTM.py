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
import time

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


def main():
    # Hyperparameters
    num_epochs = 150
    lr = 0.001
    hidden_size = 128
    num_layers = 6
    dropout = 0.15
    BATCH_SIZE = 40

    '''
    Data Select Options:
    - HR
    - IMU
    - All
    - HR + Activity
    - IMU + Activity
    - All + Activity
    '''
    data_select = 'IMU'
    activity_flag = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('*' * 50)
    if torch.cuda.is_available():  
        print('CUDA is found! Training on %s...' %torch.cuda.get_device_name(0))
    else:
        warnings.warn('CUDA not found! Training may be slow...')

    jsi_data = 'Data/JSI/1Hz'
    slade_data = 'Data/Slade/1Hz'

    source_list = getSourceList(data_select)
    data_list, target_list, activity_list = loadData(jsi_data, slade_data, source_list, activity_flag)

    test_idx = np.random.choice(len(data_list), len(data_list)//5, replace=False).tolist()
    train_idx = list(set(range(len(data_list))) - set(test_idx))
    train_data = [data_list[i] for i in train_idx]
    test_data = [data_list[i] for i in test_idx]
    train_target = [target_list[i] for i in train_idx]
    test_target = [target_list[i] for i in test_idx]
    print(np.shape(train_data[0]))
    print(np.shape(train_target[0]))

    train = Energy_Expenditure(train_data, train_target)
    test = Energy_Expenditure(test_data, test_target)

    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=pad_collate)
    test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=pad_collate)

    if data_select == 'All':
        if activity_flag == True:
            input_size = 8
        else:
            input_size = 7
        input_size = 7
    if data_select == 'IMU':
        if activity_flag == True:
            input_size = 7
        else:
            input_size = 6
    if data_select == 'HR':
        if activity_flag == True:
            input_size = 2
        else:
            input_size = 1

    model = EE_LSTM(input_dim = input_size, 
        hidden_dim = hidden_size, 
        num_layers = num_layers,
        dropout= dropout)

    model.to(device)
    model.to(torch.double) 

    criterion =  nn.MSELoss()

    optim = Adam(model.parameters(), lr=lr,betas=(0.9, 0.999))

    epoch_list = []
    train_loss_list = []
    test_loss_list = []

    start = time.time()
    for epoch in range(num_epochs):
        loss_all = 0
        loss_count = 0
        for n_batch, [in_batch, label, in_len, label_len] in enumerate(train_loader):
                in_batch, label = in_batch.to(device), label.to(device)

                output = model(in_batch)
                loss = criterion(output,label)
        
                optim.zero_grad()
                loss.backward()

                optim.step()

                # # print loss while training

                print("Epoch: [{}/{}], Batch: [{}/{}], Loss: {}".format(
                epoch, num_epochs, n_batch + 1, len(train_loader), loss.item()))
                loss_all += loss.item()
                loss_count = loss_count+1
        train_loss = loss_all / loss_count
        train_loss_list.append(train_loss)


        loss_all_test = 0
        loss_count_test = 0

        for n_batch, [in_batch, label, in_len, label_len] in enumerate(test_loader):
                in_batch, label = in_batch.to(device), label.to(device)
                output = model(in_batch)
                loss = criterion(output,label)
                optim.zero_grad()

                loss_all_test += loss.item()
                loss_count_test = loss_count_test+1
        test_loss = loss_all_test / loss_count_test
        test_loss_list.append(test_loss)

    end = time.time()
    print('Model took %d seconds to train.' %(end - start))

    plt.figure()
    plt.plot(np.linspace(0,len(train_loss_list),len(train_loss_list)),train_loss_list)
    plt.plot(np.linspace(0,len(test_loss_list),len(test_loss_list)),test_loss_list)
    plt.title('Training Loss Verse Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Loss Plot')
    plt.show()


    l1_err, l2_err = 0, 0
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for n_batch, [in_batch, label, in_len, label_len] in enumerate(test_loader):
            in_batch, label = in_batch.to(device), label.to(device)
            pred = model.test(in_batch)

            l1_err += l1_loss(pred, label).item()
            l2_err += l2_loss(pred, label).item()

        #print("Test L1 error:", l1_err)
        print("MSE error:", l2_err)

    plt.figure()
    ax = plt.subplot(1,1,1)
    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    # plot prediction vs. label
    x = np.linspace(np.min(label), np.max(label))
    y = np.linspace(np.min(label), np.max(label))
    ax = plt.subplot(1,1,1)
    ax.plot(range(np.shape(label)[1]),label[0])
    ax.plot(range(np.shape(label)[1]),pred[0])
    plt.xlabel("Time")
    plt.ylabel("Met Value")
    plt.savefig('Pred Plot')
    plt.show()

    # Save Model
    torch.save(model, 'IMU_model.pkl')

if __name__ == "__main__":
    main()