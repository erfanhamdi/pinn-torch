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


class SchrodingerNN(nn.Module):
    def __init__(self,):
        # Input layer
        self.linear_in = nn.Linear(2, 100)
        # Output layer
        self.linear_out = nn.Linear(100, 2)
        # Hidden Layers
        self.layers = nn.Sequential(
            [nn.Linear(100, 100), nn.Tanh() for i in range(5)]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.layers(x)
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
    u = model(torch.stack((x_f, t_f), axis = 1))[:, 0]
    v = model(torch.stack((x_f, t_f), axis = 1))[:, 1]
    u_t = derivative(u, t_f, order=1)
    v_t = derivative(v, t_f, order=1)
    u_xx = derivative(u, x_f, order=2)
    v_xx = derivative(v, x_f, order=2)
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u 
    return f_u, f_v

def mse_f(model, x_f, t_f):
    f_u, f_v = f(model, x_f, t_f)
    return (f_u**2 + f_v**2).pow(2).mean()

def mse_0(model, x_0, h_0):
    t_0 = torch.zeros_like(x_0)
    h_u = model(model(torch.stack((x_0, t_0), axis = 1)))[:, 0]
    h_v = model(model(torch.stack((x_0, t_0), axis = 1)))[:, 1] 
    return ((h_u-h_0[:, 0])-(h_v-h_0[:, 1])).pow(2).mean()

def mse_b(model, t_b):

    x_b_left = torch.zeros_like(t_b)-5
    h_b_left = model(torch.stack((x_b_left, t_b), axis = 1))
    h_u_b_left = h_b_left[:, 0]
    h_u_b_left_x = derivative(h_u_b_left, x_b_left, 1)
    h_v_b_left = h_b_left[:, 1]
    h_v_b_left_x = derivative(h_v_b_left, x_b_left, 1)
    
    x_b_right = torch.zeros_like(t_b)+5
    h_b_right = model(torch.stack((x_b_right, t_b), axis = 1))
    h_u_b_right = h_b_right[:, 0]
    h_u_b_right_x = derivative(h_u_b_right, x_b_right, 1)
    h_v_b_right = h_b_right[:, 1]
    h_v_b_right_x = derivative(h_v_b_right, x_b_right, 1)

    mse_drichlet = (h_u_b_left-h_u_b_right)**2+(h_v_b_left-h_v_b_right)**2
    mse_newman = (h_u_b_left_x-h_u_b_right_x)**2+(h_v_b_left_x-h_v_b_right_x)**2
    mse_total = (mse_drichlet + mse_newman).mean()
    
    return mse_total



    
# if __name__== "__main__":
