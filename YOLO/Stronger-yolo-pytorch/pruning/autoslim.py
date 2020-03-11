from pruning.BasePruner import BasePruner
from pruning.Block import *
from models.backbone.baseblock import InvertedResidual, conv_bn, sepconv_bn, conv_bias, DarknetBlock
from models.backbone.baseblock_US import bn_calibration_init
from pruning.BasePruner import BasePruner
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from pruning.Block import *
from models import *
from collections import OrderedDict
import time
from thop import clever_format
from main_dist import main_worker
import torch.multiprocessing as mp
import json


class AutoSlimPruner(BasePruner):
    def __init__(self, Trainer, newmodel, cfg):
        super().__init__(Trainer, newmodel, cfg)
        self.pruneratio = cfg.Prune.pruneratio
        self.prunestep = 32
        self.constrain = 2e9
        self.do_test=cfg.Prune.do_test
    def finetune(self,load_last,ckpt,multi_width=-1):
        self.prune(early_ret=True)
        assert ckpt is not None
        prune_iter = int(ckpt.split('/')[1].split('.')[0])

        block_channels = torch.load(ckpt)
        ## use uniform prune rate or not
        if multi_width!=-1:
            prune_iter=multi_width
            for idx, b in enumerate(self.blocks):
                if block_channels[idx] is None:
                    b.prunemask = None
                else:
                    b.prunemask = torch.arange(0, b.bnscale.shape[0]*multi_width)
        else:
            for idx, b in enumerate(self.blocks):
                if block_channels[idx] is None:
                    b.prunemask = None
                else:
                    b.prunemask = torch.arange(0, block_channels[idx]['numch'])
        self.clone_model()
        self.args.DATASET.VOC_val='test'
        self.args.EVAL.score_thres=0.2
        flops, params = self.get_flops(self.newmodel)
        # accpruned = self.test_dist(self.newmodel,cal_bn=True,valid_iter=-1)
        # print("flops:{} params:{} map:{}".format(flops, params, accpruned))
        # assert 0
        res = self.finetune_dist(savename='USprune-{}'.format(prune_iter),load_last=load_last)
        print("map finetune:{} ".format(res))
        return res
    def test(self,ckpt,multi_width=-1):
        self.prune(early_ret=True)
        assert ckpt is not None
        prune_iter = int(ckpt.split('/')[1].split('.')[0])
        block_channels = torch.load(ckpt)
        if multi_width != -1:
            prune_iter = multi_width
            for idx, b in enumerate(self.blocks):
                if block_channels[idx] is None:
                    b.prunemask = None
                else:
                    b.prunemask = torch.arange(0, b.bnscale.shape[0] * multi_width)
        else:
            for idx, b in enumerate(self.blocks):
                if block_channels[idx] is None:
                    b.prunemask = None
                else:
                    b.prunemask = torch.arange(0, block_channels[idx]['numch'])
        # #-------

        self.clone_model()
        self.args.DATASET.VOC_val='test'
        self.args.EVAL.score_thres=0.2
        res=self.test_dist(self.newmodel,cal_bn=True,valid_iter=-1,ckpt='USprune-{}'.format(prune_iter))
        print(res)
    def get_prunestep(self, numch: int):
        """ TO accelerate pruning phase
        :param numch: channel number for current layer
        :return: int
        """
        return max(int(numch * 0.1), 12)

    def prune(self, ckpt=None,early_ret=False):
        """
        :param ckpt: continue pruning with ckpt,since the search phase is slow
        :param early_ret: build the Model for other APIs
        :return: None
        """
        blocks = [None]
        name2layer = {}
        ## US model has a different architecture
        for midx, (name, module) in enumerate(self.model.named_modules()):
            if type(module) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn, conv_bias, DarknetBlock]:
                continue
            idx = len(blocks)
            if isinstance(module, DarknetBlock):
                blocks.append(DarkBlock(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, InvertedResidual):
                blocks.append(InverRes(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, conv_bn):
                blocks.append(CB(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, nn.Linear):
                blocks.append(FC(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, sepconv_bn):
                blocks.append(DCB(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, conv_bias):
                blocks.append(Conv(name, idx, [blocks[-1]], list(module.state_dict().values())))
            name2layer[name] = blocks[-1]
        self.blocks = blocks[1:]
        for b in self.blocks:
            # 两个相加的层由head部分决定
            if b.layername == 'mergelarge.conv7':
                b.inputlayer = [name2layer['headslarge.conv4']]
                b.bnscale = None
            if b.layername == 'mergemid.conv15':
                b.inputlayer = [name2layer['headsmid.conv12']]
                b.bnscale = None
        if early_ret or self.do_test:
            return
        block_channels = OrderedDict()
        for idx, b in enumerate(self.blocks):
            if b.bnscale is None:
                block_channels.update({idx: None})
            else:
                block_channels.update({
                    idx:
                        {'numch': b.bnscale.shape[0],
                         'flops': 0,
                         'params': 0,
                         'map': 0,
                         }
                })
                block_channels.update({
                    'pickidx': 0
                })
                b.prunemask = torch.arange(0, b.bnscale.shape[0])
        if ckpt:
            block_channels = torch.load(ckpt)
            for idx, b in enumerate(self.blocks):
                if block_channels[idx] is None:
                    b.prunemask = None
                else:
                    b.prunemask = torch.arange(0, block_channels[idx]['numch'])
        if ckpt is not None:
            prune_iter = int(ckpt.split('/')[1].split('.')[0]) + 1
        else:
            prune_iter = 0

        s = time.time()
        while (1):
            prune_results = []
            for idx, b in enumerate(self.blocks):
                if (block_channels[idx] == None
                        or (block_channels[idx]['numch'] - self.get_prunestep(block_channels[idx]['numch'])) <= 0
                        or block_channels[idx]['numch']<40
                ):
                    prune_results.append(-1)
                    continue
                b.prunemask = torch.arange(0, block_channels[idx]['numch'] - self.get_prunestep(
                    block_channels[idx]['numch'])).cuda()
                assert b.prunemask.shape[0] > 0
                self.clone_model()
                flops, params = self.get_flops(self.newmodel)
                accpruned = self.test_dist(self.newmodel, cal_bn=True)
                block_channels[idx]['flops'] = flops
                block_channels[idx]['params'] = params
                block_channels[idx]['map'] = accpruned
                print("flops:{} params:{} map:{}".format(flops, params, accpruned))
                prune_results.append(accpruned)
                # reset prunemask
                b.prunemask = torch.arange(0, block_channels[idx]['numch']).cuda()
            pick_idx = prune_results.index(max(prune_results))
            if block_channels[pick_idx]['flops'] < self.constrain:
                break
            block_channels[pick_idx]['numch'] -= self.get_prunestep(block_channels[pick_idx]['numch'])
            self.blocks[pick_idx].prunemask = torch.arange(0, block_channels[pick_idx]['numch']).cuda()
            print("iteration {}: prune {},current flops:{},current params:{} ,results:{},spend {}sec".format(
                prune_iter, pick_idx, block_channels[pick_idx]['flops'], block_channels[pick_idx]['params'],
                max(prune_results), round(time.time() - s)))
            block_channels['pickidx'] = pick_idx
            torch.save(block_channels, 'logs/{}.pth'.format(prune_iter))
            with open('logs/{}.json'.format(prune_iter), 'w') as f:
                json.dump(block_channels, f, indent=2)
            prune_iter += 1
