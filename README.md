# Elephant Art

## Who is he?
Elephant Art is a chinese chess engine base on convolution neural network and Monte Carlo tree search. He also support UCCI protocol.

<br>

## Warning!
Elephant Art is still pre-alpha version. Many components are not complete(Include UCCI, PGN and Perpetual Pursuit). we don't promise that he will be same format in the future.

<br>

## Build on Linux or MacOS
    $ git clone https://github.com/CGLemon/ElephantArt
    $ cd ElephantArt
    $ mkdir build && cd build
    $ cmake ..
    $ make

<br>

## More options to build
Accelerate the Network on CPU. OpenBlas is required.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

Accelerate the Network by GPU. CUDA and CUDNN are required.

    $ cmake .. -DGUP_BACKEND=CUDA
    $ cmake .. -DGUP_BACKEND=CUDA -DUSE_CUDNN=1

<br>

## Start with UCCI Interface
    $ ./Elephant -m ucci -w <weights file>
    
<br>

## Experiment Network
The network is an experiment version. The format will be changed in the future. Please check it before you use it.

https://drive.google.com/drive/folders/1NDrWH5MhAeut_sWAE55uJhvui4Hvapr5?usp=sharing

<br>

## Some Results
### Material
Following the paper, Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess. The deep mind team proposed a new way to compute the value of each piece. The basic idea is computing the effect of each piece difference. We define a feature vector
 
 <img src="https://render.githubusercontent.com/render/math?math=\LARGE F(position) = [ pawn_{diff}  ,cannon_{diff} , rook_{diff} , horse_{diff} , elephant_{diff} , advisor_{diff} , 1 ]">
 
 
 The difference is the side player's number of pieces minus opponent’s number of pieces. We define the side player winrate is 
 
  <img src="https://render.githubusercontent.com/render/math?math=\LARGE Winrate = tanh(multiply(D(x), Weights))">

We try to optimize the "Weights". Minimize the winrate lose. The "Weights" is we want to get. This method is not a good way. But we can get the roughly value in the short time. Here is the result.

| Type           | Value(Normalized) |
| :------------: | :------------:   |
| Pawn           | 1  |
| Cannon         |  3.160 |
| Rook           |  6.384 |
| Horse          |  2.782 |
| Elephant       |  1.319 |
| Advisor        |  1.124 |

<br>

### First Move
The best opening move is not most players play opening move. It is funny.

| Type                   | Move         | Probability | Winrate | Opening |
| :------------:         | :----------: | :---------: | :-----: | :-----: |
| Most players play move | h2e2, b2e2   |        60%  |  55%    | 中炮開局 |
| Best move              | g3g4, c3c4   |        15%  |  60%    | 仙人指路 |

<br>


## Reference
* UCCI protocol(chinese), https://www.xqbase.com/protocol/cchess_ucci.htm
* Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess, https://arxiv.org/abs/2009.04374
