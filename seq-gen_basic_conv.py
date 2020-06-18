import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#myfile = open('alpha_seqs.dat','r')
#Lines = myfile.readlines()
#print(Lines[0][10:])
DEVICE = torch.device('cuda:0')

def hot(line):
    """
    Given a line in a .txt or .dat file, hotencode A,C,G,Ts in the line
    """
    out = []
    encoder_dict ={'A':[1,0,0,0], 'T':[0,1,0,0], 'G':[0,0,1,0], 'C':[0,0,0,1]}
    for item in line:
        if item in encoder_dict.keys():
            out.append(encoder_dict[item])
    
    return out

def link(seq_of_seqs):
    master_list =[]
    for list1 in seq_of_seqs:
        for item in list1:
            master_list.append(item)
    
    return master_list

REBUILD_DATA = False

class SequenceDataset():
    TREE1 = 'alpha_seqs.dat'
    TREE2 = 'beta_seqs.dat'
    TREE3 = 'charlie_seqs.dat'

    TREELIST = [TREE1,TREE2,TREE3]
    LABELS = {TREE1:0, TREE2:1, TREE3:2}
    training_data = []

    def make_training_data(self):
        for label in self.LABELS:
            myfile = open(label,'r')
            Lines = myfile.readlines()
            counter = 0
            set_of_seqs =[]
            for line in Lines:
                if counter%5 != 0 and counter!=0:
                    set_of_seqs.append(hot(line))
                elif counter%5==0 and counter!=0:
                    self.training_data.append([link(set_of_seqs), np.eye(3)[self.LABELS[label]]])
                    set_of_seqs=[]

                counter+=1
        np.random.shuffle(self.training_data)
        np.save('training_Data.npy',self.training_data)

if REBUILD_DATA:
    sequence_data = SequenceDataset()
    sequence_data.make_training_data()

training_data = np.load('training_Data.npy',allow_pickle = True)
print(len(training_data[1][0]))

RETRAIN = True

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 32, 2)
        self.conv2 = nn.Conv1d(32, 64, 2)
        self.conv3 = nn.Conv1d(64, 80, 2)

        x = torch.randn(32000).view(-1,1,32000)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 200)
        self.fc2 = nn.Linear(200, 3)


    def convs(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)),2)
        x = F.max_pool1d(F.relu(self.conv2(x)),2)
        x = F.max_pool1d(F.relu(self.conv3(x)),2)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]

        return x

    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__() 

        #single hidden layer, 2 node output layer
        self.il = nn.Linear(32000,500)
        self.hl = nn.Linear(500,300)
        self.hl2 = nn.Linear(300,100)
        self.ol = nn.Linear(100,3)
    
    #feed forward
    def forward(self, x):
        x = self.il(x)
        x = self.hl(x)
        x = self.hl2(x)
        x = self.ol(x)
        return x 




if RETRAIN:

    #net = BaselineNet().to(DEVICE)
    net = Net().to(DEVICE)

    optimizer = optim.Adam(net.parameters(),lr=.001)
    loss_func = nn.MSELoss()

    X = torch.Tensor([i[0] for i in training_data]).view(-1,32000).to(DEVICE)
    y = torch.Tensor([i[1] for i in training_data]).to(DEVICE)

    VAL_PCT = .1
    val_size = int(len(X)*VAL_PCT)

    X_train = X[:-val_size]
    X_test = X[-val_size:]
    Y_train = y[:-val_size]
    Y_test = y[-val_size:]

    
    BATCH_SIZE = 8

    EPOCHS = 5

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        for i in range(0,len(X_train),BATCH_SIZE):
            batch_X = X_train[i:i+BATCH_SIZE].view(-1,1,32000) 
            batch_y = Y_train[i:i+BATCH_SIZE]

            batch_X.to(DEVICE), batch_y.to(DEVICE)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_func(outputs, batch_y)
            loss.backward()
            optimizer.step()

    print(loss)

    correct=0
    total=0





    with torch.no_grad():
        net.train(False)
        predicted_dict={0:0,1:0,2:0}
        actual_dict={0:0,1:0,2:0}
        for i in range(len(X_test)):
            real_class = torch.argmax(Y_test[i])
            actual_dict[int(real_class)] += 1
            net_out = net(X_test[i].view(-1,1,32000))[0]
            #print(net_out)
            predicted = torch.argmax(net_out)
            predicted_dict[int(predicted)]+=1
            if predicted == real_class:
                correct+=1
            total+=1

    print(f'Accurancy: {round(correct/total, 3)}')
    print(f'Predicted:{predicted_dict} , Actual:{actual_dict}')
