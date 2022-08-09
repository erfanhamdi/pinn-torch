import torch 
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial
from pyDOE import lhs
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


class SchrodingerNN(nn.Module):
    def __init__(self,):
        # Input layer
        super(SchrodingerNN, self).__init__()
        self.linear_in = nn.Linear(2, 100)
        # Output layer
        self.linear_out = nn.Linear(100, 2)
        # Hidden Layers
        self.layers = nn.ModuleList(
            [ nn.Linear(100, 100) for i in range(5) ]
        )
        # Activation function
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
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

def f(model, x_f, t_f):
    h = model(torch.stack((x_f, t_f), axis = 1))
    u = h[:, 0]
    v = h[:, 1]
    u_t = derivative(u, t_f, order=1)
    v_t = derivative(v, t_f, order=1)
    u_xx = derivative(u, x_f, order=2)
    v_xx = derivative(v, x_f, order=2)
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u 
    return f_u, f_v

def mse_f(model, x_f, t_f):
    f_u, f_v = f(model, x_f, t_f)
    return (f_u**2 + f_v**2).mean()

def mse_0(model, x_0, u_0, v_0):
    # x_0 = x_0.flatten().T
    t_0 = torch.zeros_like(x_0)
    h = model(torch.stack((x_0, t_0), axis = 1))
    h_u = h[:, 0]
    h_v = h[:, 1]
    return ((h_u-u_0)**2+(h_v-v_0)**2).mean()

def mse_b(model, t_b):
    x_b_left = torch.zeros_like(t_b)-5
    x_b_left.requires_grad = True
    h_b_left = model(torch.stack((x_b_left, t_b), axis = 1))
    h_u_b_left = h_b_left[:, 0]
    h_u_b_left_x = derivative(h_u_b_left, x_b_left, 1)
    h_v_b_left = h_b_left[:, 1]
    h_v_b_left_x = derivative(h_v_b_left, x_b_left, 1)
    
    x_b_right = torch.zeros_like(t_b)+5
    x_b_right.requires_grad = True
    h_b_right = model(torch.stack((x_b_right, t_b), axis = 1))
    h_u_b_right = h_b_right[:, 0]
    h_u_b_right_x = derivative(h_u_b_right, x_b_right, 1)
    h_v_b_right = h_b_right[:, 1]
    h_v_b_right_x = derivative(h_v_b_right, x_b_right, 1)

    mse_drichlet = (h_u_b_left-h_u_b_right)**2+(h_v_b_left-h_v_b_right)**2
    mse_newman = (h_u_b_left_x-h_u_b_right_x)**2+(h_v_b_left_x-h_v_b_right_x)**2
    mse_total = (mse_drichlet + mse_newman).mean()
    
    return mse_total

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def closure(model, optimizer, x_f, t_f, x_0, u_0, v_0, h_0, t):
    optimizer.zero_grad()
    loss = mse_f(model, x_f, t_f) + mse_0(model, x_0, u_0, v_0) + mse_b(model, t)
    loss.backward()
    global iter
    iter += 1
    print(f" iteration: {iter}  loss: {loss.item()}")
    return loss

def train(model,  x_f, t_f, x_0, u_0, v_0, h_0, t):
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
    closure_fn = partial(closure, model, optimizer, x_f, t_f, x_0, u_0, v_0, h_0, t)
    optimizer.step(closure_fn)

if __name__== "__main__":
    set_seed(42)

    # Load the data
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50
    N_b = 50
    N_f = 20000

    data = sp.loadmat('Schrodingers_Equation/NLS.mat')
    x_0 = torch.from_numpy(data['x'].astype(np.float32))
    x_0.requires_grad = True
    x_0 = x_0.flatten().T

    t = torch.from_numpy(data['tt'].astype(np.float32))
    t.requires_grad = True
    t = t.flatten().T

    h = torch.from_numpy(data['uu'])
    # Slicing the initial value of h and saving it as u_0 and v_0
    u_0 = torch.real(h)[:, 0]
    v_0 = torch.imag(h) [:, 0]
    h_0 = torch.stack((u_0, v_0), axis = 1)

    # collocation data points using latin hypercube sampling method
    c_f = lb + (ub-lb)*lhs(2, N_f)
    x_f = torch.from_numpy(c_f[:, 0].astype(np.float32))
    x_f.requires_grad = True
    t_f = torch.from_numpy(c_f[:, 1].astype(np.float32))
    t_f.requires_grad = True

    # Sampling from the initial, boundary and collocation values
    # Sample N0 points from the initial value
    # idx_0 = np.random.choice(x_0.shape[0], N0, replace = False)
    # x_0 = x_0[idx_0]
    # u_0 = u_0[idx_0]
    # v_0 = v_0[idx_0]
    # h_0 = h_0[idx_0]
    # # Sample Nb points from boundary values
    idx_b = np.random.choice(t.shape[0], N_b,replace = False )
    t_b = t[idx_b]

    model = SchrodingerNN()
    model.apply(init_weights)


    model.train()
    train(model, x_f, t_f, x_0, u_0, v_0, h_0, t)
    torch.save(model.state_dict(), 'Schrodingers_Equation/models/model_LBFGS.pt')