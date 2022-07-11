import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from main import BurgersNN

# load model
model = BurgersNN()
model.load_state_dict(torch.load('Burgers_Equation/models/model_LBFGS_shuffle_normal.pt'))
x = torch.linspace(-1, 1, 200)
t = torch.linspace( 0, 1, 100)

# x & t grids:
X, T = torch.meshgrid(x, t)

# x & t columns:
xcol = X.reshape(-1, 1)
tcol = T.reshape(-1, 1)
input = torch.cat((xcol, tcol), 1)
# one large column:
usol = model(input)

# reshape solution:
U = usol.reshape(x.numel(), t.numel())

# transform to numpy:
xnp = x.numpy()
tnp = t.numpy()
Unp = U.detach().numpy()

# plot:
fig = plt.figure(figsize=(9, 4.5))
ax = fig.add_subplot(111)

h = ax.imshow(Unp,
                interpolation='nearest',
                cmap='rainbow', 
                extent=[tnp.min(), tnp.max(), xnp.min(), xnp.max()], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h, cax=cax)
cbar.ax.tick_params(labelsize=10)
plt.show()