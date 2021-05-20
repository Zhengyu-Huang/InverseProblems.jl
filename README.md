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
    * [Bayesian inversion, Bayesian inference, and Bayesian calibration](Lectures/Bayesian.ipynb) 
    * [Markov Chain Monte Carlo method](Lectures/MonteCarlo.ipynb) 
    * [Sequential Monte Carlo method](Lectures/MonteCarlo.ipynb)
    * [Kalman filters and Gaussian approximation algorithm](Lectures/Kalman.ipynb)
    * [Unscented Kalman inversion and its variants](Lectures/Unscented.ipynb)
    * Ensemble Kalman inversion and its variants
    * [When is posterior distribution close to Gaussian](Lectures/Posterior.ipynb)
    * All models are wrong

* Linear inverse problems
    * [Well-determined, under-determined, and over-determined inverse problems](Linear/Linear-2-parameter.ipynb)
    * [Ill-conditioned matrix: inverse of Hilbert matrix](Linear/Hilbert-matrix.ipynb)
    * [High-dimensional inverse problem: 1-D elliptic equation](Linear/Elliptic.ipynb)
    * [High-dimensional inverse problem: Bernoulli random vector](Linear/Bernoulli.ipynb)

* Posterior distribution estimation
    * [Some nonlinear maps](Posterior/Nonlinear-Maps.ipynb)
    * [2-Parameter elliptic equation](Posterior/Elliptic.ipynb)
    * [1D Darcy flow](Posterior/Darcy-1D.ipynb)

* Chaotic system
    * [Chaos and butterfly effects](Chaotic/Chaos.ipynb)
    * [Lorenz63 model](Chaotic/Lorenz63.ipynb)
    * [Lorenz96 model](Chaotic/Lorenz96.ipynb)

* Structure mechanics problems
    * Damage detection of a "bridge"
    * Consitutive modeling of a multiscale fiber-reinforced plate
 
* Fluid mechanics problems
    * [2D Darcy flow](Fluid/Darcy-2D.ipynb)
    * [Navier-Stokes initial condition recovery](Fluid/Navier-Stokes.ipynb)

* Fluid structure interaction problems
    * Piston problem
        * [Receding piston (analytical solution)](FSI-Piston/Receding-Piston-Exact.ipynb)
        * [Piston system calibration](FSI-Piston/FSI.ipynb)
    * Airfoil damage detection during transonic buffeting


* Climate Modeling
    * Barotropic climate model
    * Idealized general circulation model (Held-Suarez benchmark)


## Submit an issue
You are welcome to submit an issue for any questions related to InverseProblems. 

## Here are some research papers using InverseProblem
1. Daniel Z. Huang, Tapio Schneider, and Andrew M. Stuart. "[Unscented Kalman Inversion](https://arxiv.org/pdf/2102.01580.pdf)."

2. Daniel Z. Huang, Jiaoyang Huang. "[Improve Unscented Kalman Inversion With Low-Rank Approximation and Reduced-Order Model](https://arxiv.org/pdf/2102.10677.pdf)."

3. Daniel Z. Huang, Jiaoyang Huang. "[Unscented Kalman Inversion: Efficient Gaussian Approximation to the Posterior Distribution](https://arxiv.org/pdf/2103.00277.pdf)."