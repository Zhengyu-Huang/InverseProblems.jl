#!/bin/bash
force_scale=5.0

for tid in 100 102 300
do 
    julia Data_NNPlatePull.jl $tid $force_scale 5 2 &
done

wait
