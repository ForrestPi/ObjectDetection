#NMS

## normal NMS

## soft NMS

## benchmark
cpu版 vs GPU实现加速，速度大概可以提升50X。在2080it上大概是115ms比3ms

先简要说一下CUDA编程模型：

GPU之所以能够加速，是因为并行计算，即每个线程负责计算一个数据，充分利用GPU计算核心超多（几千个）的优势。

（1）每个计算核心相互独立，运行同一段代码，这段代码称为核函数；

（2）每个核心有自己的身份id，线程的身份id是两个三维数组：（blockIdx.x，blockIdx.y，blockIdx.z）-（threadIdx.x，threadIdx.y，threadIdx.z）。

身份id被另两个三维数组grid（gridDim.x,gridDim.y,gridDim.z）和block(blockDim.x,blockDim.y,blockDim.z)确定范围。

总共有gridDim.x×gridDim.y×gridDim.z个block，

每个block有blockDim.x×blockDim.y×blockDim.z个thread。

有了线程的身份id，经过恰当的安排，让身份id（核函数可以获取）：（blockIdx.x，blockIdx.y，blockIdx.z）-（threadIdx.x，threadIdx.y，threadIdx.z）对应到一个数据，就可以实现一个线程计算一个数据，至于如何对应，开发人员得好好安排，可以 说这是CUDA开发的一个核心问题。

gridDim.x、blockIdx.x这些是核函数可以获取的，gridDim.x等于多少，调用核函数的时候就要定一下来。看代码：
```C++
  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);
```
这里的threadsPerBlock=8*8=64，

当boxes_num=12030时，DIVUP(12030, 64)=12030/64+12030%64>0=188

在调用核函数的时候，通过<<<blocks, threads>>>（#这是cu语法，不是标准C语言）把线程数量安排传递进去，核函数里就有
```C++
gridDim.x=188,gridDim.y=188,gridDim.z=1；

blockDim.x=64,blockDim.y=1,blockDim.z=1；

0<=blockIdx.x<188,

0<=blockIdx.y<188,

blockIdx.z=0,

0<=threadIdx.x<64，

threadIdx.y=threadIdx.z=0，
```
这样就启动了2,262,016个（两百多万个线程）来计算，两百多万看起来吓人，对GPU来书毫无负担！每个线程计算不超过64个值，后面再讲。

（3）这里的grid(a,b,c），block(x,y,z)值是多少，由程序设计人员根据问题来定，在调用核函数时就要确定下来，但有一个基本限制block(x,y,z)中的x×y×z<=1024（这个值随GPU版本确定，起码nvidia 1080，2080都是这样）；

（4）block中的线程每32个thread为一束，绝对同步：比如if-else语句，这32个线程中有的满足if条件，有的满足else。满足else的那部分线程不能直接进入，而是要等满足if的那部分线程运行完毕才进入else部分，而满足if的那部分线程现在也不能结束，而是要等else部分线程运行完毕，大家才能同时结束。for语句也是一样。因此GPU计算尽可能不要有分支语句。

不是说不能用if和for，该用还得用，用的时候要知道付出的代价。否则实现了减速都不知道为了啥。

不同的线程束之间不同步，如果同步需要请__syncthreads();

如果设置block(1)，即一个block只安排一个线程呢？事实上GPU还是要启动32个线程，另外31个陪跑。

因此block(x,y,z)中的x×y×z应该为32的倍数。不过32×32=1024了。

（5）要并行计算，前提是数据之间没有相互依赖，有前后依赖的部分只能放在同一个核函数里计算；

先看控制部分代码，这部分做的事情就是：

1,在GPU上分配内存，把数据传到GPU

2，调用核函数，计算mask；

3，把数据传回来，

4，根据mask把获取保留下来的候选框。