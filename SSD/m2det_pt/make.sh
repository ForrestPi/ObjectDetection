#!/usr/bin/env bash
cd ./utils/

CUDA_PATH=/usr/local/cuda/

sudo rm -r build/
rm ./nms/*.so
python3 build.py build_ext --inplace

cd ..
