# Burgers' Equation 
## Intro
This equation describes the movement of a viscous fluid with one spatial (x) and one temporal (t) dimension like a thin ideal pipe with fluid running through it. it describes the speed of fluid at each location along the pipe as time progresses
```math
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
```
## Data Prepration
The boundary condition was taken to be 
```math
\begin{aligned}
u &= 0  \;\;@ x &= 1\\
u &= 0  \;\;@ x &= -1\\
u &= -sin(\pi x)  \;\;@ t &= 0
\end{aligned}
```
## Parameter Study
<!-- 1. The first test was done using the proposed Number of N_u and N_f but with full batch Adam optimizer and Xavier Normal initialization weights with a lr = 0.001 for 20k epochs

![adam](/Burgers_Equation/figures/adam_xavier_uniform_001_bias.png)
2. The second test was done using full batch L-BFGS and Normal Xavier Uniform initialization after 6899 epochs and the input data was shuffled.
![lbfgs](/Burgers_Equation/figures/lbfgs_xavier_uniform_shuffle.png)

3. The third test has the same conditions to the second test but without shuffling the input data after 4763 epochs
![logisticReg](/Burgers_Equation/figures/lbfgs_xavier_uniform_no_shuffle.png)

4. 4679 Epochs, Shuffle, Xavier Normal -->

|        Adam, 20k   | LBFGS, 6.9k, Xavier-Uniform, shuffle | LBFGS, 4.7k, Xavier-Normal, shuffle  |  LBFGS, 4.7k, Xavier-Uniform, no-shuffle  |
|:----------:|:-------------:|:-------------:|:-------------:|
|![](/Burgers_Equation/figures/adam_20k.png)|![](/Burgers_Equation/figures/lbfgs_xavier_uniform_shuffle.png)|![](/Burgers_Equation/figures/lbfgs_xavier_normal_shuffle.png)|![](/Burgers_Equation/figures/lbfgs_xavier_uniform_no_shuffle.png)


## Acknowledgement
1. data preprocessing was taken from this implementation [PINN-Burgers](https://github.com/EdgarAMO/PINN-Burgers)
2. plotting functions were taken from [PINN](https://github.com/maziarraissi/PINNs)