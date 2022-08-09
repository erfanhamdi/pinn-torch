# Schrodingers' Equation 
## Intro
The one-dimensional nonlinear Schrodinger equation is a classical field equation that is used to study quantum mechanical systems, including nonlinear wave propagation in optical fibers and/or waveguides, Bose-Einstein condensates, and plasma waves. The nonlinear 1-D Schrodinger equation is:
```math
i\frac{\partial h}{\partial t} + 0.5\frac{\partial^2 h}{\partial x^2} + \abs{h}^2h = 0
```
## Data Prepration
In this case a periodic boundary condition was used
```math
\begin{aligned}
h(0, x) &= 2sech(x)
h(t, -5) &= h(t, 5)
h_x(t, -5) &= h_x(t, 5)
\end{aligned}
```
## Parameter Study
This case was solved using [L-BFGS](https://erfanhamdi.github.io/blog_posts/l-bfgs/lbfgs.html) optimization method.

|        Adam, 20k   | LBFGS, 6.9k, Xavier-Uniform, shuffle | LBFGS, 4.7k, Xavier-Normal, shuffle  |  LBFGS, 4.7k, Xavier-Uniform, no-shuffle  |
|:----------:|:-------------:|:-------------:|:-------------:|
|![](/Burgers_Equation/figures/adam_20k.png)|![](/Burgers_Equation/figures/lbfgs_xavier_uniform_shuffle.png)|![](/Burgers_Equation/figures/lbfgs_xavier_normal_shuffle.png)|![](/Burgers_Equation/figures/lbfgs_xavier_uniform_no_shuffle.png)


## Acknowledgement
1. plotting functions were taken from [PINN](https://github.com/maziarraissi/PINNs)