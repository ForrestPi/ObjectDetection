import os
import torch
from trainers.base_trainer import BaseTrainer
from models.backbone.baseblock import InvertedResidual, conv_bn, sepconv_bn,conv_bias,DarknetBlock
from pruning.Block import *
from main_dist import main_worker
import torch.multiprocessing as mp
from utils.util import pick_avail_port
class BasePruner:
    def __init__(self,trainer:BaseTrainer,newmodel,cfg):
        self.model=trainer.model
        self.newmodel=newmodel
        self.trainer=trainer
        self.blocks=[]
        self.pruneratio = 0.1
        self.args=cfg
    def get_flops(self,model):
        from thop import clever_format, profile
        input = torch.randn(1, 3, 512, 512).cuda()
        flops, params = profile(model, inputs=(input,), verbose=False)
        return flops,params
    def prune(self,ckpt=None):
        blocks = [None]
        name2layer = {}
        for midx, (name, module) in enumerate(self.model.named_modules()):
            if type(module) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn, conv_bias,DarknetBlock]:
                continue
            idx = len(blocks)
            if isinstance(module,DarknetBlock):
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
            if b.layername == 'mergelarge.conv7':
                b.inputlayer=[name2layer['headslarge.conv4']]
            if b.layername == 'headsmid.conv8':
                b.inputlayer.append(name2layer[self.args.Prune.bbOutName[1]])
            if b.layername == 'mergemid.conv15':
                b.inputlayer=[name2layer['headsmid.conv12']]
            if b.layername == 'headsmall.conv16':
                b.inputlayer.append(name2layer[self.args.Prune.bbOutName[0]])
    def test(self,newmodel=True,validiter=-1,cal_bn=False):
        raise NotImplementedError
    def test_dist(self,model,cal_bn,valid_iter=-1,ckpt=''):
        self.args.do_test=True
        self.args.EXPER.resume=ckpt
        # self.args.ngpu=2
        parent_conn, child_conn = mp.Pipe()
        avail_port=pick_avail_port()
        # mp.spawn(main_worker,nprocs=self.args.ngpu,args=(self.args.ngpu,self.args,self.newmodel,child_conn,avail_port,cal_bn))
        mp.spawn(main_worker,nprocs=self.args.ngpu,args=(self.args.ngpu,self.args,model,child_conn,avail_port,cal_bn,valid_iter))
        return parent_conn.recv()[0]
    def finetune(self,load_last,epoch=10):
        raise NotImplementedError
    def finetune_dist(self,savename='',load_last=False):
        assert savename!=''
        self.trainer.model=self.newmodel
        self.best_mAP=0
        self.args.do_test=False
        self.args.EXPER.resume= savename if load_last else ''
        self.args.EXPER.save_ckpt=savename
        # self.args.ngpu=2
        parent_conn, child_conn = mp.Pipe()
        avail_port=pick_avail_port()
        mp.spawn(main_worker, nprocs=self.args.ngpu, args=(self.args.ngpu, self.args, self.newmodel, child_conn,avail_port))
        return parent_conn.recv()[0]
    def clone_model(self):
        blockidx = 0
        for name, m0 in self.newmodel.named_modules():
            if type(m0) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn,conv_bias,DarknetBlock]:
                continue
            block = self.blocks[blockidx]
            curstatedict = block.statedict
            if (len(block.inputlayer) == 1):
                if block.inputlayer[0] is None:
                    inputmask = torch.arange(block.inputchannel)
                else:
                    inputmask = block.inputlayer[0].outmask
            elif (len(block.inputlayer) == 2):
                first = block.inputlayer[0].outmask
                second = block.inputlayer[1].outmask
                second+=block.inputlayer[0].outputchannel
                second=second.to(first.device)
                inputmask=torch.cat((first,second),0)
            else:
                raise AttributeError
            if isinstance(block,DarkBlock):
                assert len(curstatedict)==(1+4+1+4)
                block.clone2module(m0,inputmask)
            if isinstance(block, CB):
                # conv(1weight)->bn(4weight)->relu
                assert len(curstatedict) == (1 + 4)
                block.clone2module(m0, inputmask)
            if isinstance(block, DCB):
                # conv(1weight)->bn(4weight)->relu
                assert len(curstatedict) == (1 + 4 + 1 + 4)
                block.clone2module(m0, inputmask)
            if isinstance(block, InverRes):

                # dw->project or expand->dw->project
                assert len(curstatedict) in (10, 15)
                block.clone2module(m0, inputmask)
            if isinstance(block, FC):
                block.clone2module(m0)
            if isinstance(block, Conv):
                block.clone2module(m0,inputmask)

            blockidx += 1
            if blockidx > (len(self.blocks) - 1): break