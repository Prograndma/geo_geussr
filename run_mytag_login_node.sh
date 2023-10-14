#!/bin/bash

# some helpful debugging options
set -e
set -u

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load jq zstd pigz parallel libnvidia-container enroot                                                                                                   

CONTAINER_NAME="mycontainer"

# Check if container already exists using enroot list
if ! enroot list | grep -q "^${CONTAINER_NAME}\$"; then
    enroot create --force --name $CONTAINER_NAME ${HOME}/mytag.sqsh
fi

# run a shell 
# Ensures that the run_training doesn't try using cuda
enroot start  --mount /lustre/scratch/usr/${USER}:/home/${USER}/compute --rw \
       --mount ${HOME}/CS674/project1:/app/CS674/project1 --rw \
       --mount ${HOME}/CS674/project1/base_ViT:/app/CS674/project1/base_ViT --rw \
       --mount ${HOME}/CS674/project1/checkpoints:/app/CS674/project1/checkpoints --rw \
       --mount ${HOME}/CS674/project1/dataset:/app/CS674/project1/dataset --rw \
       --mount ${HOME}/CS674/project1/processor:/app/CS674/project1/processor --rw \
       $CONTAINER_NAME \
       bash -c "export CUDA_VISIBLE_DEVICES='' && ./run_training.sh --access_internet=1 --max_train_steps=1
