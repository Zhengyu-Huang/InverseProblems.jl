#!/bin/bash


SOWER_EXECUTABLE=/home/dzhuang/frg/sower/build/bin/sower 
MATCHER_EXECUTABLE=/home/dzhuang/frg/matcher/build/bin/matcher  
PARTDMESH_EXECUTABLE=/home/dzhuang/frg/metis/build/bin/partdmesh

NS=35
NP=$NS
NC=$NS

# Decompose fluid mesh
$PARTDMESH_EXECUTABLE sources/domain.top $NS

# Run Matcher to match the structure surface and structure file
$MATCHER_EXECUTABLE sources/embeddedSurf.top sources/agard.matcher -e 1 -p 8 -l 2  -output data/agard

# Run Sower to pre-process the fluid mesh
$SOWER_EXECUTABLE -fluid -mesh sources/domain.top -match data/agard.match.fluid -dec sources/domain.top.dec.$NS -cpu $NP -cluster $NC -output data/agard


