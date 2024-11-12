#!/bin/bash

# Get the absolute path of the project root
export GOTORCH_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add libtorch to library path
export LD_LIBRARY_PATH=$GOTORCH_ROOT/cgotorch/linux/libtorch/lib:$LD_LIBRARY_PATH

# Add cgotorch to library path
export LD_LIBRARY_PATH=$GOTORCH_ROOT/cgotorch:$LD_LIBRARY_PATH
