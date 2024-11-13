#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()

    if ! command_exists make; then
        missing_deps+=("make")
    fi

    if ! command_exists clang++; then
        missing_deps+=("clang++")
    fi

    if ! command_exists go; then
        missing_deps+=("go")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Error: Missing required dependencies:"
        printf '%s\n' "${missing_deps[@]}"
        exit 1
    fi
}

# Check dependencies first
check_dependencies

pushd $DIR

OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
CXX="g++"
LIBTORCH_DIR=""
GLIBCXX_USE_CXX11_ABI="1"
LOAD="force_load"
LIB_SUFFIX="so"
INSTALL_NAME=""
CUDA_FLAGS=""

version_ge() {
    local ver1_major=$(echo "$1" | cut -d. -f1)
    local ver1_minor=$(echo "$1" | cut -d. -f2)
    local ver2_major=$(echo "$2" | cut -d. -f1)
    local ver2_minor=$(echo "$2" | cut -d. -f2)
    if [ "$ver1_major" -gt "$ver2_major" ]; then
        return 0
    elif [ "$ver1_major" -eq "$ver2_major" ]; then
        if [ "$ver1_minor" -ge "$ver2_minor" ]; then
            return 0
        fi
    fi
    return 1
}

function build_linux_no_cuda() {
    CXX="clang++"
    LIBTORCH_DIR="linux/libtorch"
    GLIBCXX_USE_CXX11_ABI="1"
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        echo "Downloading CPU libtorch 2.5.1..."
        wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
        if [ $? -ne 0 ]; then
            echo "Failed to download libtorch"
            exit 1
        fi
        echo "Extracting libtorch..."
        unzip -qq -o libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip -d linux
        if [ $? -ne 0 ]; then
            echo "Failed to extract libtorch"
            exit 1
        fi
        rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip
    fi
}

if [[ "$OS" == "linux" ]]; then
    if [[ "$ARCH" =~ arm* ]]; then
        echo "Building for Raspbian ..."
        CXX="g++"
        LIBTORCH_DIR="rpi/libtorch"
        if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
            curl -LsO 'https://github.com/ljk53/pytorch-rpi/raw/master/libtorch-rpi-cxx11-abi-shared-1.6.0.zip'
            unzip -qq -o libtorch-rpi-cxx11-abi-shared-1.6.0.zip -d rpi
        fi
    elif command -v /usr/local/cuda/bin/nvcc >/dev/null 2>&1; then
        CXX="clang++"
        CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep release | grep -Eo "[0-9]+.[0-9]+" | head -1)

        # Check if CUDA_RUNTIME environment variable is set
        if [ -z "${CUDA_RUNTIME}" ]; then
            echo "Error: CUDA_RUNTIME environment variable is not set"
            echo "Please set it to your CUDA installation path using:"
            echo "export CUDA_RUNTIME=/path/to/cuda"
            echo "For example: export CUDA_RUNTIME=/usr/local/cuda"
            exit 1
        fi

        #CUDA_FLAGS="-DWITH_CUDA -I${CUDA_RUNTIME}/include"
        CUDA_FLAGS="-DWITH_CUDA -I${CUDA_RUNTIME}/include -I${CUDA_RUNTIME}"

        if version_ge "$CUDA_VERSION" "12.4"; then
            echo "Building for Linux with CUDA $CUDA_VERSION (>= 12.4)"
            LIBTORCH_DIR="linux/libtorch"
            CUDA_ZIP="libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip"
            if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
                if [[ ! -f "$DIR/$CUDA_ZIP" ]]; then
                    echo "Downloading CUDA 12.4 libtorch..."
                    wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
                fi
                echo "Extracting libtorch"
                unzip -qq -o libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip -d linux
            fi
        elif [[ "$CUDA_VERSION" == "12.1" ]]; then
            echo "Building for Linux with CUDA 12.1"
            LIBTORCH_DIR="linux/libtorch"
            if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
                wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip
                unzip -qq -o libtorch-cxx11-abi-shared-with-deps-2.5.1+cu121.zip -d linux
            fi
        elif [[ "$CUDA_VERSION" == "11.8" ]]; then
            echo "Building for Linux with CUDA 11.8"
            LIBTORCH_DIR="linux/libtorch"
            if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
                wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip
                unzip -qq -o libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip -d linux
            fi
        else
            echo "Unsupported CUDA version: $CUDA_VERSION"
            echo "Falling back to CPU version..."
            build_linux_no_cuda
        fi
    else
        echo "No CUDA detected, building CPU version..."
        build_linux_no_cuda
    fi
elif [[ "$OS" == "darwin" ]]; then
    echo "Building for macOS ..."
    CXX="clang++"
    LIBTORCH_DIR="macos/libtorch"
    LIB_SUFFIX="dylib"
    INSTALL_NAME="-install_name @rpath/\$@"
    LOAD="all_load"
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        curl -LsO https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip
        unzip -qq -o libtorch-macos-1.6.0.zip -d macos
    fi
fi

rm -f libtorch
ln -s ${LIBTORCH_DIR} libtorch

echo "[DEBUG] Make command parameters:"
echo "CXX=$CXX"
echo "LIB_SUFFIX=$LIB_SUFFIX"
echo "INSTALL_NAME=$INSTALL_NAME"
echo "LIBTORCH_DIR=$LIBTORCH_DIR"
echo "GLIBCXX_USE_CXX11_ABI=$GLIBCXX_USE_CXX11_ABI"
echo "LOAD=$LOAD"
echo "CUDA_FLAGS=$CUDA_FLAGS"

set -o xtrace
make CXX="$CXX" \
     LIB_SUFFIX="$LIB_SUFFIX" \
     INSTALL_NAME="$INSTALL_NAME" \
     LIBTORCH_DIR="$LIBTORCH_DIR" \
     GLIBCXX_USE_CXX11_ABI="$GLIBCXX_USE_CXX11_ABI" \
     LOAD="$LOAD" \
     CUDA_FLAGS="$CUDA_FLAGS" \
     -f Makefile -j
popd
