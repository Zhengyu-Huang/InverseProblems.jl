# Airfoil damage detection 

This example demonstrates how to calibrate black-box model with off-the-shelf software , specifically the [AERO suite](https://bitbucket.org/%7Bfe0b433b-144f-4d1b-affe-009d9d15c999%7D/) developed in Farhat Research Group.

We consider a challenging real-world fluid structure interaction (FSI) problem associated with a damaged AGARD wing undergoing transonic buffet. The forward model is based on the embedded boundary method, and the damage field is infered from the displacement measurements of the sensors the wing.


## Forward problem

The structure solver is *aeros* and the fluid solver is *aerof*. 
To start the FSI simulation, we need to generate fluid/structure solver input files. 

    * Generate fluid mesh file `domain.top` in `./sources` folder with the `Simplex.py` 

    ```console
    python Simplex.py
    ```
    
    * Generate structure embedded surface file `sources/embeddedSurf.top`
    
    * Generate structure input file `simulations/agard.unsteady`, which include the strucutre mesh `simulations/agard.fem.common` and material propery file `simulations/agard.fem.composite` (this needs to be calibrated)
    
    * Generate fluid input files, including `agard.steady` for steady simulation and `agard.unsteady` for unsteady FSI simulation
    
    
Then we need preprocess these mesh files for parallel computing and coupling between structure embedded surface and structure model.
    
    ```console
    ./prepro.sh
    ```
    
Finally, we can run the simulations (both steady and unsteady simulations) in the `./simulations` folder 

`mpirun -n 35 $AEROF_EXECUTABLE agard.steady |& tee steady.log`

`mpirun -n 35 $AEROF_EXECUTABLE agard.unsteady : -n 1 $AEROS_EXECUTABLE agard.fem.unsteady |& tee unsteady.log`

A more convenient way is to write these command lines in the bash file, run the simulations

` ./run.steady`

` ./run.unsteady`

which also include command lines to loading libraries and postprocessing, the `.exo` file for paraview visualization.


## Inverse problem

Unscented Kalman inversion routine each time modifies the input file `simulations/agard.fem.composite`, calls the AERO suite ` ./run.unsteady`, and reads the output file of the AERO suite to get the observation.
For the current setup, we run the code interactively.

 * Generate initial flow condition by runing steady simulation 
 
 `./run.steady`

 * Generate synthetic observation data
 
 `julia FSI_UKI_Ref.jl`
 
 * Move reference observation to the reference fold 
 
 `mkdir results.1/reference`
 `mv results.1/AGARD.disp* results.1/reference/.`

* Start calibration 

  `julia FSI_UKI.jl`