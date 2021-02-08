#!/bin/bash
force_scale=5.0

for tid in 100 101 102 103 104 105 106 200 201 202 203 204 205 206 300
do 
    julia Data_NNPlatePull.jl $tid $force_scale 5 2 &
done

wait
