from models import *
from trainers import *
import json
from yacscfg import _C as cfg
import os
from torch import optim
import argparse
import numpy as np
from thop import clever_format,profile
from pruning import SlimmingPruner,AutoSlimPruner,l1normPruner
from mmcv.runner import load_checkpoint
import torch
from main_dist import main_worker
def main(args):
    assert args.Prune.pruner!=''
    model = eval(cfg.MODEL.modeltype)(cfg=args.MODEL).cuda().eval()
    newmodel = eval(cfg.MODEL.modeltype)(cfg=args.MODEL).cuda().eval()
    optimizer = optim.Adam(model.parameters(),lr=args.OPTIM.lr_initial)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.OPTIM.milestones, gamma=0.1)
    _Trainer = eval('Trainer_{}'.format(args.DATASET.dataset))(args=args,
                       model=model,
                       optimizer=optimizer,
                       lrscheduler=scheduler
                       )

    pruner=eval(args.Prune.pruner)(_Trainer,newmodel,cfg=args)
    pruner.prune(ckpt=None)
    ##---------count op
    input=torch.randn(1,3,512,512).cuda()
    flops, params = profile(model, inputs=(input, ),verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    flopsnew, paramsnew = profile(newmodel, inputs=(input, ),verbose=False)
    flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")
    print("flops:{}->{}, params: {}->{}".format(flops,flopsnew,params,paramsnew))
    if not args.Prune.do_test:
        ## For AutoSlim, specify the ckpt
        if args.Prune.pruner=='AutoSlimPruner':
            bestfinetune= pruner.finetune(load_last=False,ckpt='logs/265.pth')
        else:
            bestfinetune=pruner.finetune(load_last=False)
        print("finetuned map:{}".format(bestfinetune))
    else:
        ## For AutoSlim, specify the ckpt
        if args.Prune.pruner=='AutoSlimPruner':
            bestfinetune=pruner.test(ckpt='logs/265.pth',)
        else:
            bestfinetune=pruner.test()
        print("finetuned map:{}".format(bestfinetune))

  #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        # default='configs/strongerv3_US_prune.yaml'
        default = 'configs/strongerv3_prune.yaml'
        # default = 'configs/strongerv2_prune.yaml'
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # do not freeze
    # cfg.freeze()
    main(args=cfg)
