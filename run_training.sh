#!/bin/bash



/opt/conda/bin/accelerate launch train_geo.py \
  $1 \
  $2
