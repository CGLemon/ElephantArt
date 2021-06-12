# Elephant Art

## Who is he?

Elephant Art 是一個基於神經網路和蒙地卡羅樹搜索的象棋引擎。並且也支援 UCCI 協議。

Elephant Art is a chinese chess engine base on convolution neural network and Monte Carlo tree search. He also support UCCI protocol.

<br>

## Warning!
Elephant Art is still pre-alpha version. Many components are not complete(Include UCCI, PGN and Perpetual Pursuit). We don't promise that he will be same format in the future.

<br>

## Build on Linux or MacOS
    $ git clone https://github.com/CGLemon/ElephantArt
    $ cd ElephantArt
    $ mkdir build && cd build
    $ cmake ..
    $ make

<br>

## More options to build

Unlike the other chinese chess engine based on network. Elephant Art doesn't need any backend. But you still accelerate the network by third party blas library. Here is it.

Accelerate the network on CPU. OpenBlas is required.

    $ cmake .. -DBLAS_BACKEND=OPENBLAS

Accelerate the network by GPU. CUDA is required. It will be faster than cuDNN in only one batch size.

    $ cmake .. -DGUP_BACKEND=CUDA

Accelerate the network by GPU. CUDA and CUDNN are required. It will be faster than CUDA-only in large batch size.

    $ cmake .. -DGUP_BACKEND=CUDA -DUSE_CUDNN=1

<br>

## Some options to start the Elephant Art

Here are some useful options you can use.

    --weights, -w: Load the network weights file.

    $ ./Elephant -w <weights file>


    --playouts, -p: Set the playouts limit. Bigger is stronger.

    $ ./Elephant -p 1600


    --threads, -t: Set the search threads. Bigger is faster.

    $ ./Elephant -t 4


    --batchzie, -b: Set the network batch size. Bigger is faster. But the batch size is better small than threads.

    $ ./Elephant -b 2


    --analysis-verbose: Output more search verbose.

    $ ./Elephant --analysis-verbose


    --mode, -m: Start with different modes. Here is UCCI mode.

    $ ./Elephant -m ucci


    Exmaple:

    $ ./Elephant -m ucci -w <weights file> -t 4 -b 2 -p 1600

<br>

## Experiment Network
The network is an experiment version. The format will be changed in the future. Please check it before you use it.

https://drive.google.com/drive/folders/1NDrWH5MhAeut_sWAE55uJhvui4Hvapr5?usp=sharing

The training pgn data is from the website, WXF webside and dpxq webside. You can download the pgn data from this side if you need it [象棋棋譜](https://github.com/CGLemon/chinese-chess-PGN).

<br>

## Some Results

### Strength
The supervised learning is not quite good. It will make too many stupid moves, kill itself move or meanless move. But he can still beat the most amateur players. May reach Taiwanese 1~2 dan level.

<br>

### Material
Following the paper, Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess. The deep mind team proposed a new way to compute the value of each piece. The basic idea is computing the effect of each piece difference. We define a feature vector
 
 <img src="https://render.githubusercontent.com/render/math?math=\LARGE F(position) = [ pawn_{diff}  ,cannon_{diff} , rook_{diff} , horse_{diff} , elephant_{diff} , advisor_{diff} , 1 ]">
 
 
 The difference is the side player's number of pieces minus opponent’s number of pieces. We define the side player winrate is 
 
  <img src="https://render.githubusercontent.com/render/math?math=\LARGE Winrate = tanh(multiply(F(x), Weights))">

We try to optimize the "Weights". Minimize the winrate lose. The "Weights" means value of each piece. This method is not a good way. But we can get the roughly value in the short time. Here is the result.

| Type           | Value(Normalized) |
| :------------: | :---------------: |
| Pawn           | 1                 |
| Cannon         | 3.160             |
| Rook           | 6.384             |
| Horse          | 2.782             |
| Elephant       | 1.319             |
| Advisor        | 1.124             |

<br>

### First Move
The best opening move is not most players play opening move. It is funny.

| Type                   | Move         | Probability | Winrate | Opening |
| :------------:         | :----------: | :---------: | :-----: | :-----: |
| Most players play move | h2e2, b2e2   | 60%         | 55%     | 中炮開局 |
| Best move              | g3g4, c3c4   | 15%         | 60%     | 仙人指路 |

<br>

### Policy Accuracy

| Type   | Size                   | Data Set   | Accuracy | Date      |
| :----: | :--------------------: | :--------: | :------: | :-------: |
| Resnet | 4 blocks x 256 filters | validation | 38.5%    | 2021-5/23 |
| Resnet | 4 blocks x 256 filters | training   | 54%      | 2021-5/23 |

<br>

### Reinforcement Learning
coming soon...

<br>

## Reference
* UCCI protocol(chinese), https://www.xqbase.com/protocol/cchess_ucci.htm
* Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess, https://arxiv.org/abs/2009.04374

<br>

## LICENSE
GNU GPL version 3 section 7
