# InverseProblems


InverseProblems is a

* educational **Inverse Problem** or **Numerical Method** library. 
The goal is to provide students with a light-weighted code to explore these areas 
and interactive lectures with amazing [Jupyter Notebook](https://jupyter.org/).
* benchmark repository originally designed to test unscented Kalman inversion and other **derivative-free inverse methods**. 
The goal is to provide reseachers with access to various inverse problems, 
while enabling researchers to quickly and easily develop and test novel inverse methods.

## Code Struction
* All the inverse methods are in *Inversion* folder
* Each other folder contains one category of inverse problems

## Tutorial
Let's start! (⚠️ under construction)

* What are inverse problems, why are they important?

* Inverse methods
    * Markov Chain Monte Carlo method
    * Sequential Monte Carlo method
    * Unsented Kalman inversion and its variants
    * Ensemble Kalman inversion and its variants

* Linear inverse problems
    * [Well-determined, under-determined, and over-determined inverse problems](Linear/Linear-2-parameter.ipynb)
    * [Ill-conditioned matrix: inverse of Hilbert matrix](Linear/Hilbert-matrix.ipynb)
    * [High-dimensional inverse problem :1-D elliptic equation](Linear/Elliptic.ipynb)
    * [High-dimensional inverse problem :Bernoulli random vector](Linear/Bernoulli.ipynb)

* Bayesian approach
    * Bayesian inversion, Bayesian inference, and Bayesian calibration 
    * All models are wrong
    * Some nonlinear maps
    * Elliptic equation

* Chaotic system (Runge Kutta method)
    * Chaos and butterfly effects
    * Lorenz63 model
    * Lorenz96 model

* Structure mechanics problems (Finite element method)
    * Damage detection of a "bridge"
    * Consitutive modeling of a multiscale fiber-reinforced plate
 
* Fluid mechanics problems (Finite difference and spectral methods)
    * 1D Darcy flow
    * 2D Darcy flow
    * Navier-Stokes initial condition recovery 

* Climate Modeling (Spectral dynamical core)
    * Barotropic climate model
    * Idealized general circulation model (Held-Suarez benchmark)


## Submit an issue
You are welcome to submit an issue for any questions related to InverseProblems. 

## Here are some research papers using InverseProblem
1. Daniel Z. Huang, Tapio Schneider, and Andrew M. Stuart. "[Unscented Kalman Inversion](https://arxiv.org/pdf/2102.01580.pdf)."

2. Daniel Z. Huang, Jiaoyang Huang. "[Improve Unscented Kalman Inversion With Low-Rank Approximation and Reduced-Order Model](https://arxiv.org/pdf/2102.10677.pdf)."
