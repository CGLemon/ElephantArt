# Elephant Art

## Who is he?
Elephant Art is a chinese chess engine base on convolution neural network and Monte Carlo tree search. He also support UCCI protocol.

## Warning!
Elephant Art is still pre-alpha version. Many components are not complete(Include UCCI, PGN and Perpetual Pursuit). we don't promise that he will be same format in the future.

## Build on Linux or MacOS
    $ git clone https://github.com/CGLemon/ElephantArt
    $ cd ElephantArt
    $ mkdir build && cd build
    $ cmake ..
    $ make
    
## Other Option To Build
Accelerate the Network on CPU. OpenBlas is required.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

Accelerate the Network by GPU. CUDA and CUDNN are required.

    $ cmake .. -DGUP_BACKEND=CUDA
    $ cmake .. -DGUP_BACKEND=CUDA -DUSE_CUDNN=1

## Start With UCCI Interface
    $ ./Elephant -m ucci -w <weights file>
    
## Experiment Network
The network is a experiment version. The format will be changed in the future. Please check it before you use it.

https://drive.google.com/drive/folders/1NDrWH5MhAeut_sWAE55uJhvui4Hvapr5?usp=sharing

## Reference
* UCCI protocol(chinese): https://www.xqbase.com/protocol/cchess_ucci.htm
