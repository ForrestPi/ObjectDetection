
## DIoU
https://github.com/Zzh-tju/DIoU    
## KLloss
https://github.com/yihui-he/KL-Loss    
## Focal-Loss
https://github.com/hedgefair/Focal-Loss-Pytorch    
## AP-loss
https://github.com/cccorn/AP-loss    
## GHM_Loss
https://github.com/ForrestPi/GHM_Loss    
## AnchorLoss
https://github.com/slryou41/AnchorLoss    
## SCELoss
https://github.com/HanxunHuangLemonBear/SCELoss-Reproduce    


IoU：使用最广泛的检测框loss。

GIoU：2019年CVPR    Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression

DIoU和CIoU：2020年AAAI  Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression

下面我们直接一句话总结一下这四种算法的优缺点：
1. IoU算法是使用最广泛的算法，大部分的检测算法都是使用的这个算法。

2. GIoU考虑到，当检测框和真实框没有出现重叠的时候IoU的loss都是一样的，因此GIoU就加入了C检测框（C检测框是包含了检测框和真实框的最小矩形框），这样就可以解决检测框和真实框没有重叠的问题。但是当检测框和真实框之间出现包含的现象的时候GIoU就和IoU loss是同样的效果了。

3. DIoU考虑到GIoU的缺点，也是增加了C检测框，将真实框和预测框都包含了进来，但是DIoU计算的不是框之间的交并，而是计算的每个检测框之间的欧氏距离，这样就可以解决GIoU包含出现的问题。

4. CIoU就是在DIoU的基础上增加了检测框尺度的loss，增加了长和宽的loss，这样预测框就会更加的符合真实框。

这些只是我看的重点，详细的还需要大家去阅读论文看一下效果。
