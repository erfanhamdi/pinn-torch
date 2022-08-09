import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True
import torch

from main import SchrodingerNN

from PIL import Image

def make_gif():
    frames = [Image.open(image) for image in sorted(Path("output/").iterdir(), key=os.path.getmtime)[1:]]
    frame_one = frames[0]
    frame_one.save("figures/convergence.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=1)


# load model
model = SchrodingerNN()
x = torch.linspace(-5, 5, 200)
t = torch.linspace( 0, 1.5, 100)

# x & t grids:
X, T = torch.meshgrid(x, t)

# x & t columns:
xcol = X.reshape(-1, 1)
tcol = T.reshape(-1, 1)
input = torch.cat((xcol, tcol), 1)

paths = sorted(Path("models/").iterdir(), key=os.path.getmtime)
# print(paths)
for idx, model_add in enumerate(paths[-2:-1]):
    
    fig = plt.figure(figsize=(9, 4.5))
    # model iteration number 
    iter = model_add.stem.split("_")[-1]
    plt.title(r'$|h(t, x)|$', fontsize=16)
    plt.xlabel(r'$t$', fontsize=14)
    plt.ylabel(r'$x$', fontsize=14)

    model.load_state_dict(torch.load(model_add))

    # input variables to the model
    usol = model(input)
    u_magnitude = (usol[:, 0]**2 + usol[:, 1]**2)**0.5

    # reshape solution:
    U = u_magnitude.reshape(x.numel(), t.numel())

    # transform to numpy:
    xnp = x.numpy()
    tnp = t.numpy()
    Unp = U.detach().numpy()

    # plot:
    ax = fig.add_subplot(111)

    h = ax.imshow(Unp,
                    interpolation='nearest',
                    cmap='YlGnBu', 
                    extent=[tnp.min(), tnp.max(), xnp.min(), xnp.max()], 
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    plt.savefig("output/" + str(model_add).split('/')[1].split('.')[0] + '.png')
    plt.close()

# make_gif()