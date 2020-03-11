from pruning.BasePruner import BasePruner
from pruning.Block import *
from models.backbone.baseblock import InvertedResidual, conv_bn, sepconv_bn, conv_bias,DarknetBlock


class SlimmingPruner(BasePruner):
    def __init__(self, Trainer, newmodel, cfg,savebn=''):
        super().__init__(Trainer, newmodel,cfg)
        self.pruneratio = cfg.Prune.pruneratio
        self.savebn=savebn
    def test(self,newmodel=True,validiter=-1,cal_bn=False):
        res=self.test_dist(self.newmodel,cal_bn=False,valid_iter=-1,ckpt='ft-{}'.format(self.pruneratio))
        return res
    def prune(self,ckpt=None):
        super().prune()
        # gather BN weights
        bns = []
        maxbn=[]
        blacklist = [b.layername for b in self.blocks if 'residual_downsample' in b.layername]
        for b in self.blocks:
            if b.bnscale is not None and b.layername not in blacklist:
                bns.extend(b.bnscale.tolist())
                maxbn.append(b.bnscale.max().item())
        bns = torch.Tensor(bns)
        y, i = torch.sort(bns)


        # import numpy as np
        # np.save('v2bn.npy',y)
        # assert 0
        prunelimit=(y==min(maxbn)).nonzero().item()/len(bns)
        print("prune limit: {}".format(prunelimit))
        if self.pruneratio>prunelimit:
            raise AssertionError('prune ratio bigger than limit')
        thre_index = int(bns.shape[0] * self.pruneratio)
        thre = y[thre_index]
        thre = thre.cuda()
        pruned_bn = 0
        for b in self.blocks:
            if isinstance(b, CB):
                ## for darknet pruing, residual_downsample's output must be kept
                if 'residual_downsample' in b.layername:
                    mask = torch.ones_like(b.bnscale)
                    b.prunemask = torch.arange(b.bnscale.shape[0])
                else:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, InverRes):
                if b.numlayer == 3:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                    print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, DCB):
                mask = b.bnscale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b,DarkBlock):
                mask = b.bnscale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
        self.clone_model()
        print("Slimming Pruner done")
    def finetune(self,load_last,epoch=10):
        res=self.finetune_dist(savename='ft-{}'.format(self.pruneratio))
        return res