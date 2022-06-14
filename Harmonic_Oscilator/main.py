import yaml

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import torch
import torch.nn as nn

with open("Harmonic_Oscilator/config.yml") as f:
    config = yaml.safe_load(f)

def set_seed(seed: int = 42):
    '''
    Seeding the random variables for reproducibility
    ''' 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PhysicsNN(nn.Module):
    def __init__(self):
        super(PhysicsNN, self).__init__()
        # Input layer
        self.linear_in = nn.Linear(1, 32)
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

def derivative(model: PhysicsNN, x_f: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    This function calculates the derivative of the model at x_f
    """
    dy = model(x_f)
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x_f, grad_outputs = torch.ones_like(x_f), create_graph=True, retain_graph=True
        )[0]
    return dy
          
def u_function(model: PhysicsNN, x: torch.Tensor) -> torch.Tensor:
    """
    This function evaluates the model on the input x
    """
    return model(x)

def f(model: PhysicsNN, x_f: torch.Tensor, m: float, miu: float, k: float) -> torch.Tensor:
    """
    This function evaluates the physics governing the model on the input x_f
    """
    u = u_function(model, x_f)
    dudx = derivative(model, x_f, order = 1)
    d2udx2 = derivative(model, x_f, order = 2)
    f = d2udx2 + miu*dudx + k*u
    return f

def loss_function(model: PhysicsNN, x_u: torch.Tensor, x_f: torch.Tensor, y_u: torch.Tensor) -> torch.Tensor:
    m = config["m"]
    miu = config["miu"]
    k = config["k"]
    # Loss associated with the physics governing the model
    MSE_f = f(model, x_f, m, miu, k).pow(2).mean()*1e-4
    # Loss associated with the boundary conditions and the data
    MSE_u = (u_function(model, x_u.float())-y_u).pow(2).mean()
    return MSE_f + MSE_u

def train(model: PhysicsNN, x_u: torch.Tensor, x_f: torch.Tensor, y_u: torch.Tensor):
    """
    This function trains the model on the input data
    """
    epoch = config["epoch"]
    lr = config["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

    # loading the training data on the boundary
    x_data = np.loadtxt("Harmonic_Oscilator/data/x_data.csv")
    y_data = np.loadtxt("Harmonic_Oscilator/data/y_data.csv")
    x_u = torch.tensor(x_data, requires_grad=True).reshape(x_data.shape[0], 1)
    y_u = torch.tensor(y_data, requires_grad=True).reshape(x_data.shape[0], 1)
    
    x_f = torch.linspace(0, 1, 100).reshape(-1, 1)
    x_f.requires_grad = True

    # Instantiating the model
    model = PhysicsNN()

    # Training the model on the boundary data and the collocation data
    model, loss_list = train(model, x_u, x_f, y_u)

    # Plotting the results
    y_f = model(x_f).detach().numpy()
    x_f = x_f.detach().numpy()
    plt.plot(x_f, y_f, label = "Predicted")
    plt.title("Harmonic Oscillator")
    plt.ylabel(r"$f(x)$")
    plt.xlabel(r"$x$")
    plt.savefig("Harmonic_Oscilator/predicted.png", dpi = 100)
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

