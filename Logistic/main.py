from torch import nn
import torch
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

class Logistic_PINN(nn.Module):
    '''
    The nn Class containing the Architecture of our PINN
    '''
    def __init__(self):
        super(Logistic_PINN, self).__init__()
        # Linear MLP with 5 hidden Layers each 10 nodes
        self.layer_in = nn.Linear(1, 10)
        self.layer_out = nn.Linear(10, 1)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(10, 10) for _ in range(5)]
        )
        # Tanh activation function
        self.act = nn.Tanh()
    
    def forward(self, t):
        out_0 = self.layer_in(t)
        for layer in self.hidden_layers:
            out_0 = self.act(layer(out_0))
        out_1 = self.layer_out(out_0)
        return out_1

def f(nn, t_train):
    '''
    This function evaluates the PINN model of the Logistic Model
    '''
    return nn(t_train)

def df(nn, t_train, order=1):
    '''
    Differentiating function using pytorch autograd
    '''
    df_eval = nn(t_train)
    for d in range(order):
        df_eval = torch.autograd.grad(
            df_eval, t_train, grad_outputs=torch.ones_like(t_train), create_graph=True, retain_graph=True
        )[0]
    return df_eval

def loss_func(nn_instance, t_train):
    '''
    Combination of the Logistic Eqn and the BC as the loss of the function
    '''
    # Evaluation of the function itself
    interior_loss = df(nn_instance, t_train) - R * t_train*(1-t_train)
    
    # Evaluation of the boundary condition 
    boundary = torch.Tensor([T0])
    boundary.requires_grad = True
    boundary_loss = f(nn_instance, boundary) - F0
    
    # Combination of the two
    total_loss = interior_loss.pow(2).mean() + boundary_loss**2
    
    return total_loss

def train_func(nn_instance, t_train, epochs=20_000, lr=0.01):
    '''
    Training function for the PINN
    '''

    # Initializing the Adam optimizer
    optimizer = torch.optim.Adam(nn_instance.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_func(nn_instance, t_train)
        loss_list = []
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    return nn_instance, loss_list

def set_seed(seed=42):
    '''
    Seeding the random variables for reproducibility
    ''' 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == "__main__":
    
    # Setting the seed for the random vars
    set_seed()

    # Colocation Points
    t_loss = torch.linspace(0, 1, steps = 100, requires_grad=True)
    t_loss = t_loss.reshape(t_loss.shape[0], 1)

    t_train = torch.linspace(0, 1, steps = 10, requires_grad=True)
    t_train = t_train.reshape(t_train.shape[0], 1)

    # Boundary and Initial Conditions
    T0 = 0
    F0 = 1

    # Funciton Parameters
    R = 1

    # Instantiate the NN Model
    nn_instance = Logistic_PINN()

    # Train the Model
    nn_trained, loss_list = train_func(nn_instance, t_train, epochs=20_000)
    
    # Function Evaluation
    fig, ax = plt.subplots()

    # Plotting the Results
    f_final_training = f(nn_trained, t_train)
    f_final = f(nn_trained, t_loss)
    
    ax.scatter(t_train.detach().numpy(), f_final_training.detach().numpy(), label="Training points", color="red")
    ax.plot(t_loss.detach().numpy(), f_final.detach().numpy(), label="NN final solution")
    plt.xlabel(r't')
    plt.ylabel(r'$f(x)$')
    plt.savefig("Logistic/figures/logistic_pinn.png")
    plt.show()
