# Schrodingers' Equation 
## Intro
The one-dimensional nonlinear Schrodinger equation is a classical field equation that is used to study quantum mechanical systems, including nonlinear wave propagation in optical fibers and/or waveguides, Bose-Einstein condensates, and plasma waves. The nonlinear 1-D Schrodinger equation is:
```math
i\frac{\partial h}{\partial t} + 0.5\frac{\partial^2 h}{\partial x^2} + |h|^2h = 0
```
## Data Prepration
In this case a periodic boundary condition was used
```math
\begin{aligned}
h(0, x) &= 2sech(x)\\
h(t, -5) &= h(t, 5)\\
h_x(t, -5) &= h_x(t, 5)
\end{aligned}
```
## Parameter Study
This case was solved using [L-BFGS](https://erfanhamdi.github.io/blog_posts/l-bfgs/lbfgs.html) optimization method.

|        Convergence animation   | LBFGS, 6.9k, Loss 1.47e-5 |
|:----------:|:-------------:|
|![](/Schrodingers_Equation/figures/convergence.gif)|![](/Schrodingers_Equation/figures/model_LBFGS_6960.png)|

## Acknowledgement
1. plotting functions were taken from [PINN](https://github.com/maziarraissi/PINNs)