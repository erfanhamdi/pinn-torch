# Burgers' Equation 
## Intro
This equation describes the movement of a viscous fluid with one spatial (x) and one temporal (t) dimension like a thin ideal pipe with fluid running through it. it describes the speed of fluid at each location along the pipe as time progresses
```math
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
```
## Data Prepration
The boundary condition was taken to be 
```math
u = 0 @ x = 1
u = 0 @ x = -1
u = 0 @ t = 0
```
## Parameter Study
1. The first test was done using the proposed Number of N_u and N_f but with full batch Adam optimizer and Xavier Normal initialization weights with a lr = 0.001 for 20k epochs
![[adam_xavier_uniform_001_bias.png]]
2. The second test was done using full batch L-BFGS and Normal Xavier Uniform initialization after 4763 epochs and the input data was shuffled.![[Screen Shot 2022-07-11 at 2.18.50 PM.png]]
3. The third test has the same conditions to the second test but without shuffling the input data after 4763 epochs![[Screen Shot 2022-07-11 at 2.47.54 PM.png]] 
4. 

## Citations
data preprocessing was taken from this implementation [PINN-Burgers](https://github.com/EdgarAMO/PINN-Burgers)
