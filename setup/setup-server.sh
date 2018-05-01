#!/bin/bash
conda create -n ml python=3.6.3 anaconda;
source activate ml;
pip install tensorflow-gpu==1.5 numpy pandas matplotlib keras;
pip install --upgrade numpy;
