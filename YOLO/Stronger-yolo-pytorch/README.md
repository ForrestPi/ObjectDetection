# Strongeryolo-pytorch 

## Introduction
 I reimplemented with Pytorch and continue improving yolov3 with latest papers.  
This project will also try out some model-compression approaches(e.g. channel-pruning).  

## Environment
python3.6, pytorch1.2(1.0+ should be ok), ubuntu14/16/18 tested.


## Improvement with latest papers(Using StrongerV3 as baseline)
### All results all tested with 544*544 and threshold 0.1
|model|mAP50|mAP75|configs|baseline|
| ------ | ------ | ------ |------ |------ |
|baseline(with GIOU)|79.6 |43.4|strongerv3.yaml|-|
|+ [kl loss&&varvote](https://github.com/yihui-he/KL-Loss)|78.9|49.2 |strongerv3_kl.yaml|strongerv3.yaml|  
|+ [ASFF](https://github.com/ruinmessi/ASFF)|80.6|46.6 |strongerv3_asff.yaml|strongerv3.yaml|
|+ All improvement|81.1|53.0 |strongerv3_all.yaml|strongerv3.yaml|

Note:  
1.Set EVAL.varvote=True to enable varvote in KL-loss. According to the paper, kl-loss(and varvote) can strongly boost the performance of mAP75(or higher), but decrease mAP50 slightly.  
2.The details(e.g. channel number) of ASFF module is not completely the same as the original implementation.  
3.The **All** version including other small tricks like removing relu in detection head. Check config file for details. 
## Performance on VOC2007 Test(mAP) after pruning
|Model| Pruner|Backbone|mAP(before/after prune) | Flops(G)| Params(M)|
| ------ | ------ | ------ | ------ |------ |------ |
strongerv3|/|Mobilev2|79.6|4.33|6.775|
strongerv3-(40% pruned)|Slimming |Mobilev2|77.4/76.9 |2.64|2.75|
strongerv3-(pruned)|AutoSlim |Mobilev2|78.5/75.0|2.64|3.34|
| ************* |*************|************* |*************|************* |************* |
strongerv2| /|Darknet53|80.2|49.8|61.6|
strongerv2-(70% pruned)|Slimming |Darknet53|78.1/77.1 |38.9|16.8|  

Note:  
1.Tuning _C.Prune.sr can get better prune ratio, I picked the official number 0.01.  


## Supported backbone
- [x] MobileV2(Pruning suppoted)
- [x] DarkNet(Pruning supported)
...

## Supported Pruner
- [x] [l1-norm pruner](https://arxiv.org/abs/1608.08710)
- [x] [Slimming pruner](https://arxiv.org/abs/1708.06519)
- [x] [AutoSlim](https://arxiv.org/abs/1903.11728) (Update 2020-3-7)

## Reference
[Stronger-Yolo](https://github.com/Stinky-Tofu/Stronger-yolo)  
[focal-loss](https://arxiv.org/abs/1708.02002)  
[kl-loss](https://github.com/yihui-he/KL-Loss)
[ASFF](https://github.com/ruinmessi/ASFF)
