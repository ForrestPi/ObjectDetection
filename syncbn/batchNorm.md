
https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

```python
def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape
    #step1: calculate mean
    mu = 1./N * np.sum(x, axis = 0)
    #step2: subtract mean vector of every trainings example
    xmu = x - mu
    #step3: following the lower branch - calculation denominator
    sq = xmu ** 2
    #step4: calculate variance
    var = 1./N * np.sum(sq, axis = 0)
    #step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)
    #step6: invert sqrtwar
    ivar = 1./sqrtvar
    #step7: execute normalization
    xhat = xmu * ivar
    #step8: Nor the two transformation steps
    gammax = gamma * xhat
    #step9
    out = gammax + beta
    #store intermediate
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    return out, cache

```


```python
def batchnorm_backward(dout, cache):
    #unfold the variables stored in cache
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    #get the dimensions of the input/output
    N,D = dout.shape
    #step9
    dbeta = np.sum(dout, axis=0)
    dgammax = dout #not necessary, but more understandable
    #step8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma
    #step7
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar
    #step6
    dsqrtvar = -1. /(sqrtvar**2) * divar
    #step5
    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
    #step4
    dsq = 1. /N * np.ones((N,D)) * dvar
    #step3
    dxmu2 = 2 * xmu * dsq
    #step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
    #step1
    dx2 = 1. /N * np.ones((N,D)) * dmu
    #step0
    dx = dx1 + dx2
    return dx, dgamma, dbeta


```

https://www.aiuai.cn/aifarm1206.html
```python
class SyncBatchNorm(_BatchNorm):
    #Cross-GPU Synchronized Batch normalization (SyncBN)
    def __init__(self, 
                 num_features, 
                 eps=1e-5, 
                 momentum=0.1, 
                 sync=True, 
                 activation="none", 
                 slope=0.01,
                 inplace=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=True)
        self.activation = activation
        self.inplace = False if activation == 'none' else inplace
        #self.inplace = inplace
        self.slope = slope
        self.devices = list(range(torch.cuda.device_count()))
        self.sync = sync if len(self.devices) > 1 else False
        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]
        # running_exs
        #self.register_buffer('running_exs', torch.ones(num_features))

    def forward(self, x):
        # Resize the input to (B, C, -1).
        input_shape = x.size()
        x = x.view(input_shape[0], self.num_features, -1)
        if x.get_device() == self.devices[0]:
            # Master mode
            extra = {
                "is_master": True,
                "master_queue": self.master_queue,
                "worker_queues": self.worker_queues,
                "worker_ids": self.worker_ids
            }
        else:
            # Worker mode
            extra = {
                "is_master": False,
                "master_queue": self.master_queue,
                "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
            }
        if self.inplace:
            return inp_syncbatchnorm(
                x, 
                self.weight, 
                self.bias, 
                self.running_mean, 
                self.running_var,
                extra, 
                self.sync, 
                self.training, 
                self.momentum, 
                self.eps,
                self.activation, 
                self.slope).view(input_shape)
        else:
            return syncbatchnorm(
                x, 
                self.weight, 
                self.bias, 
                self.running_mean, 
                self.running_var,
                extra, 
                self.sync, 
                self.training, 
                self.momentum, 
                self.eps,
                self.activation, 
                self.slope).view(input_shape)

    def extra_repr(self):
        if self.activation == 'none':
            return 'sync={}'.format(self.sync)
        else:
            return 'sync={}, act={}, slope={}, inplace={}'.format(
                self.sync, self.activation, self.slope, self.inplace
            )
```