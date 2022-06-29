import yaml

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pyDOE import lhs
from burgers_utils import newfig, savefig

import torch
import torch.nn as nn



def set_seed(seed: int = 42):
    '''
    Seeding the random variables for reproducibility
    ''' 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BurgersNN(nn.Module):
    def __init__(self,):
        super(BurgersNN, self).__init__()
        # Input layer
        self.linear_in = nn.Linear(2, 32)
        # Output layer
        self.linear_out = nn.Linear(32, 1)
        # Hidden layers
        self.layers = nn.ModuleList(
            [nn.Linear(32, 32) for i in range(2)]
        )
        # Activation function
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    This function calculates the derivative of the model at x_f
    """
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs = torch.ones_like(dy), create_graph=True, retain_graph=True
        )[0]
    return dy

def u_function(model: BurgersNN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    This function evaluates the model on the input x
    """
    model_input = torch.stack((x, t), axis = 1)
    return model(model_input)

def f(model, x_f, t_f):
    u = u_function(model, x_f, t_f)
    u_t = derivative(u, t_f, order=1)
    u_x = derivative(u, x_f, order=1)
    u_xx = derivative(u, x_f, order=2)
    f = u_t + u.T*u_x - (0.01/np.pi)*u_xx
    return f

def loss_function(model: BurgersNN, x_u: torch.Tensor, x_f: torch.Tensor, t_f: torch.Tensor, t_u: torch.Tensor, y_u: torch.Tensor) -> torch.Tensor:
    """
    This function evaluates the physics governing the model on the input x_f
    """
    u = u_function(model, x_f, t_f)
    MSE_f = f(model, x_f, t_f).pow(2).mean()
    MSE_u = (u_function(model, x_u, t_u)-y_u).pow(2).mean()
    return MSE_f + MSE_u

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train(model: BurgersNN, u_params: torch.Tensor, x_f: torch.Tensor,
        t_f: torch.Tensor, y_u: torch.Tensor,
        epochs: int = 100, lr: float = 0.001) -> BurgersNN:
    """
    This function trains the model on the input data
    """
    # Setting the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    x_u = u_params[:, 0]
    t_u = u_params[:, 1]
    for i in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, x_u, x_f, t_f, t_u, y_u)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if i % 1_000 == 0:
            print(f'Epoch {i}: {loss.item()}')
    return model, loss_list




if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    # Load config file
    # with open("config.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    
    # Number of collocation points
    N_f = 10_000
    N_u = 100

    # Collocation points
    x_f = torch.linspace(0, 1, N_f, requires_grad=True)
    t_f = torch.linspace(0, 1, N_f, requires_grad=True)

    # boundary conditions 
    # at x = -1
    data = scipy.io.loadmat('Burgers_Equation/burgers_shock.mat')
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    x_u = torch.Tensor(X_u_train)
    x_u.requires_grad = True
    u_train = u_train[idx,:]
    u_train = torch.Tensor(u_train)
    u_train.requires_grad = True
    
    # Model instantiation
    model = BurgersNN()
    model.apply(init_weights)
    # Training
    model, loss_list = train(model, x_u, x_f, t_f, u_train, epochs=5_000, lr=0.001)

    # save the model
    torch.save(model.state_dict(), 'Burgers_Equation/model.pt')
    
    

