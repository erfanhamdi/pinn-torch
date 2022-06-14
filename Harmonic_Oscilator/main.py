
from this import d
from turtle import forward
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def set_seed(seed=42):
    '''
    Seeding the random variables for reproducibility
    ''' 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def oscillator(d, w0, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

class PhysicsNN(nn.Module):
    def __init__(self):
        super(PhysicsNN, self).__init__()
        self.linear_in = nn.Linear(1, 32)
        self.linear_out = nn.Linear(32, 1)
        self.layers = nn.ModuleList(
            [nn.Linear(32, 32) for i in range(2)]
        )
        self.act = nn.Tanh()
    def forward(self, x):
        x = self.linear_in(x)
        x = self.act(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x

def derivative(model, x, order):
    dy = model(x)
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs = torch.ones_like(x), create_graph=True, retain_graph=True
        )[0]
    return dy
          
def u_function(model, x):
    return model(x)

def f(model, x_f, m, miu, k):
    u = u_function(model, x_f)
    dudx = derivative(model, x_f, order = 1)
    d2udx2 = derivative(model, x_f, order = 2)
    f = d2udx2 + miu*dudx + k*u
    return f

def loss_function(model, x_u, x_f, y_u):
    MSE_f = f(model, x_f, m = 1, miu = 4, k = 400).pow(2).mean()*1e-4
    MSE_u = (u_function(model, x_u.float())-y_u).pow(2).mean()
    return MSE_f + MSE_u

def train(model, x_u, x_f, y_u, epoch = 100):
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    loss_list = []
    for epoch in range(epoch):
        optimizer.zero_grad()
        loss = loss_function(model, x_u, x_f, y_u)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 1_000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    return model, loss_list

if __name__ == "__main__":
    # Seeding random variables
    set_seed(42)

    d, w0 = 2, 20

    # get the analytical solution over the full domain
    x = torch.linspace(0,1,500).view(-1,1)
    y = oscillator(d, w0, x).view(-1,1)
    # print(x.shape, y.shape)

    # slice out a small number of points from the LHS of the domain
    x_u = x[0:200:20]
    y_u = y[0:200:20]

    # loading the training data on the boundary
    x_data = np.loadtxt("Harmonic_Oscilator/x_data.csv")
    y_data = np.loadtxt("Harmonic_Oscilator/y_data.csv")
    x_u = torch.tensor(x_data, requires_grad=True).reshape(x_data.shape[0], 1)
    y_u = torch.tensor(y_data, requires_grad=True).reshape(x_data.shape[0], 1)
    
    x_f = torch.linspace(0, 1, 30).reshape(-1, 1)
    x_f.requires_grad = True

    # Instantiating the model
    model = PhysicsNN()

    # Training the model on the boundary data and the collocation data
    model, loss_list = train(model, x_u, x_f, y_u, epoch = 20_000)

    # Plotting the results
    y_f = model(x_f).detach().numpy()
    x_f = x_f.detach().numpy()
    plt.plot(x_f, y_f, label = "Predicted")
    plt.show()

    # x_further = torch.linspace(0, 10, 100).reshape(-1, 1)
    # x_further.requires_grad = True
    # y_further = model(x_further).detach().numpy()
    # x_further = x_further.detach().numpy()
    # plt.plot(x_further, y_further, label = "Further")
    # plt.show()
    # x_test = torch.linspace(0,1,500).view(-1,1)
    # y_test = model(x_test)
    # plt.figure()
    # plt.plot(x_test, y_test, label="Predicted solution")
    # plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
    # plt.legend()
    # plt.show()

