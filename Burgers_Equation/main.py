import yaml
from random import uniform
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

iter = 0

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
        self.linear_in = nn.Linear(2, 20)
        # Output layer
        self.linear_out = nn.Linear(20, 1)
        # Hidden layers
        self.layers = nn.ModuleList(
            [nn.Linear(20, 20) for i in range(8)]
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
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def closure(model: BurgersNN, optimizer, X_u_train: torch.Tensor, X_f_train:torch.Tensor, Y_u_train: torch.Tensor) -> torch.Tensor:
    """
    In order to use the LBFGS optimizer, we need to define a closure function. This function is called by the optimizer
    and the optimizer contains the inner loop for the optimization and it continues until the tolerance is met.
    """
    x_u = X_u_train[:, 0]
    t_u = X_u_train[:, 1]
    x_f = X_f_train[:, 0]
    t_f = X_f_train[:, 1]
    y_u = Y_u_train
    optimizer.zero_grad()
    loss = loss_function(model, x_u, x_f, t_f, t_u, y_u)
    loss.backward()
    global iter
    iter += 1
    print(f" iteration: {iter}  loss: {loss.item()}")
    return loss

def train(model, X_u_train, X_f_train, u_train):
    # Initialize the optimizer
    optimizer = torch.optim.LBFGS(model.parameters(),
                                    lr=1,
                                    max_iter=50000,
                                    max_eval=50000,
                                    history_size=50,
                                    tolerance_grad=1e-05,
                                    tolerance_change=0.5 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")

    # the optimizer.step requires the closure function to be a callable function without inputs
    # therefore we need to define a partial function and pass it to the optimizer
    closure_fn = partial(closure, model, optimizer, X_u_train, X_f_train, u_train)
    optimizer.step(closure_fn)

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    nu = 0.01 / np.pi         # constant in the diff. equation
    N_u = 100                 # number of data points in the boundaries
    N_f = 10000               # number of collocation points

    # X_u_train: a set of pairs (x, t) located at:
        # x =  1, t = [0,  1]
        # x = -1, t = [0,  1]
        # t =  0, x = [-1, 1]
    x_upper = np.ones((N_u//4, 1), dtype=float)
    x_lower = np.ones((N_u//4, 1), dtype=float) * (-1)
    t_zero = np.zeros((N_u//2, 1), dtype=float)

    t_upper = np.random.rand(N_u//4, 1)
    t_lower = np.random.rand(N_u//4, 1)
    x_zero = (-1) + np.random.rand(N_u//2, 1) * (1 - (-1))

    # stack uppers, lowers and zeros:
    X_upper = np.hstack( (x_upper, t_upper) )
    X_lower = np.hstack( (x_lower, t_lower) )
    X_zero = np.hstack( (x_zero, t_zero) )

    # each one of these three arrays haS 2 columns, 
    # now we stack them vertically, the resulting array will also have 2 
    # columns and 100 rows:
    X_u_train = np.vstack( (X_upper, X_lower, X_zero) )

    # shuffle X_u_train:
    index = np.arange(0, N_u)
    np.random.shuffle(index)
    X_u_train = X_u_train[index, :]
    
    # make X_f_train:
    X_f_train = np.zeros((N_f, 2), dtype=float)
    for row in range(N_f):
        x = uniform(-1, 1)  # x range
        t = uniform( 0, 1)  # t range

        X_f_train[row, 0] = x 
        X_f_train[row, 1] = t

    # add the boundary points to the collocation points:
    X_f_train = np.vstack( (X_f_train, X_u_train) )

    # make u_train
    u_upper =  np.zeros((N_u//4, 1), dtype=float)
    u_lower =  np.zeros((N_u//4, 1), dtype=float) 
    u_zero = -np.sin(np.pi * x_zero)  

    # stack them in the same order as X_u_train was stacked:
    u_train = np.vstack( (u_upper, u_lower, u_zero) )

    # match indices with X_u_train
    u_train = u_train[index, :]
    # Model instantiation
    model = BurgersNN()
    model.apply(init_weights)
    # Training
    X_u_train = torch.from_numpy(X_u_train).requires_grad_(True).float()
    X_f_train = torch.from_numpy(X_f_train).requires_grad_(True).float()
    u_train = torch.from_numpy(u_train).requires_grad_(True).float()

    model.train()
    train(model, X_u_train, X_f_train, u_train)
    # save the model
    torch.save(model.state_dict(), 'Burgers_Equation/model_LBFGS_shuffle_normal.pt')
    
    

