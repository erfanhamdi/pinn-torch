# Logistic Equation
## Intro
Let  𝐾  represent the carrying capacity for a particular organism in a given environment, and let  𝑟  be a real number that represents the growth rate. The function  𝑃(𝑡)  represents the population of this organism as a function of time  𝑡 , and the constant  𝑃_0  represents the initial population (population of the organism at time  𝑡=0 ). Then the logistic differential equation is
```math
\frac{\partial{P}}{\partial{t}} - rP(1-\frac{P}{K}) = 0
```
## Boundary Conditions
P(0) = 1

## Results
![logisticReg](/Logistic/figures/logistic_pinn.png)

## Acknowledgement
- This wonderful post on Towards DataScience by Mario Dagrada : [Introduction to Physics Informed Neural Networks.](https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4)
- This entry on Logistic Equation on Mathematics LibreText : [The Logistic Equation](https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/08%3A_Introduction_to_Differential_Equations/8.4%3A_The_Logistic_Equation)
