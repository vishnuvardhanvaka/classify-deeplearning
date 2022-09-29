import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import math
import  pandas as pd

class iris(Dataset):
    def __init__(self):
        xy=np.loadtxt('datasets/iris.csv',delimiter=',',dtype=str,skiprows=1)
        print(xy)
        self.x=torch.tensor(np.array(xy[:,:4],dtype=np.float32))
        y=(xy[:,[4]])
        mapping={'"Setosa"':0,'"Versicolor"':1,'"Virginica"':2}
        for i in range(len(y)):
            st=y[i,0].item()
            y[i,0]=mapping[st]
        self.y=torch.LongTensor(np.array(y,dtype=np.int32))
        
        self.n_samples=self.x.shape[0]
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples
dataset=iris()
batch_size=50
#spilit dataset 
features,labels=dataset[:]
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2)
y_train=y_train.reshape(1,120)

print(y_train[0])
#dataloader
dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

#model
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(4,5)
        self.fc2=nn.Linear(5,6)
        self.fc3=nn.Linear(6,3)
    def forward(self,x):
        af=nn.ReLU()
        x=af(self.fc1(x))
        x=af(self.fc2(x))
        x=self.fc3(x)
        return x
model=ANN()

#settings
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs=200
iterations=math.ceil(len(dataset)/batch_size)

#training
for epoch in range(epochs):
    y_hat=model.forward(x_train)
    loss=loss_fn(y_hat,y_train[0])
    if epoch%10==0:
        print(loss,epoch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#testing
pred=[]
act=[]
accuracy=[]
for i in range(len(x_test)):
    y_hat=model.forward(x_test[i])
    index=y_hat.argmax()
    pred.append(index.item())
    act.append(y_test[i].item())
    if index.item()==y_test[i].item():
        accuracy.append(1)
    else:
        accuracy.append(0)
table=pd.DataFrame({'predicted':pred,'actual':act,'accuracy':accuracy})
print(table)






