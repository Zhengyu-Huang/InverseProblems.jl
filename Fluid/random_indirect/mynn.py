import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# A neural network with u_n， θ_c
# u_d = K(θ_c) u_n
# u_d(x) = \int K(x, y, θ_c) u_n(y) dy
class DirectKernelNet(nn.Module):

    def __init__(self, N_θ):
        super(DirectKernelNet, self).__init__()
        self.N_θ = N_θ
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_θ + 2, 20)
        self.fc2 = nn.Linear(20, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 20)
        self.fc5 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x



# # A neural network with u_n， θ_c
# # u_d = K(θ_c) u_n
# # u_d(x) = \int K(x, y, θ_c) u_n(y) dy
# class InDirectKernelNet(nn.Module):

#     def __init__(self, N_θ, N_x, L):
#         super(InDirectKernelNet, self).__init__()
#         self.N_θ = N_θ
#         self.N_θ = N_θ
#         # an affine operation: y = Wx + b
        
#         self.fc1 = nn.Linear(N_θ + 2, 20)
#         self.fc2 = nn.Linear(20, 50)
#         self.fc3 = nn.Linear(50, 50)
#         self.fc4 = nn.Linear(50, 20)
#         self.fc5 = nn.Linear(20, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x


# preprocess the training data 

class DirectData(Dataset):

    def __init__(self, X, y):
        
        self.X = X if torch.is_tensor(X) else torch.from_numpy(X)
        self.y = y if torch.is_tensor(y) else torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data(prefix="", seeds = []):
    # concatenate data
    θs, κs = [], []

    if not seeds:
        θ = np.load(prefix+"random_direct_theta.npy")   
        κ = np.load(prefix+"random_direct_K.npy")
    else:
        # load data 
        for seed in seeds:
            print("load random_direct_theta."+str(seed)+".npy and random_direct_K."+str(seed)+".npy")
            θs.append(np.load(prefix+"random_direct_theta."+str(seed)+".npy"))
            κs.append(np.load(prefix+"random_direct_K."+str(seed)+".npy"))

        θ = np.concatenate(θs, axis = 0)   
        κ = np.concatenate(κs, axis = 2)


    # N_data, N_θ =  θ.shape
    # N_x, N_y, N_data = κ.shape
    return θ, κ

def build_bases(κ, N_trunc=-1, acc=0.9999):

    N_x, N_y, N_data = κ.shape

    data = κ.reshape((-1, N_data))

    # svd bases
    u, s, vh = np.linalg.svd(np.transpose(data))
    
    if N_trunc < 0:
        s_sum_tot = sum(s)
        s_sum = 0.0
        for i in range(N_data):
            s_sum += s[i]
            if s_sum > acc*s_sum_tot:
                break
        N_trunc = i+1
    print("N_trunc = ", N_trunc)



    scale = np.average(s[:N_trunc])
    data_svd = u[:, 0:N_trunc] * s[:N_trunc]/scale
    bases = vh[0:N_trunc, :]*scale

    return data_svd, bases, N_trunc


