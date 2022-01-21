import torch
import random
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import numpy as np
from .segLoss import SegMetrics


class SegNet(BaseModel):
    def __init__(self, opt):
        super(SegNet, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.criterionL1 = torch.nn.L1Loss()
        self.num_classes = 8
        self.L1_Lambda = 40.
        weight = torch.ones(self.num_classes).to(self.device)
        # weight
        weight = np.array([0.3725, 0.0548, 0.0114, 0.0210, 0.0112, 0.3457, 0.0859, 0.0975], dtype=np.float32)
        weight = 1. / np.log(weight*100 + 1.02)
        weight = torch.tensor(weight).to(self.device)
        self.criterionBCE = torch.nn.NLLLoss2d(weight)
        self.netG = networks.define_G(input_nc=3, output_nc=self.num_classes, ngf=32, netG='ERFNet', norm=opt.norm,
                                      use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        self.model_names = ['G']

        if self.isTrain:
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        if self.isTrain:
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, samples):
        self.real_DS = samples['DS'].to(self.device)
        self.real_DI = samples['DI'].to(self.device)
        self.real_SS = samples['SS'].to(self.device)
        self.real_SI = samples['SI'].to(self.device)

    def forward(self):
        b, c, h, w = self.real_DS.size() #b, 8, 512,512
        self.fake_SS = self.netG(self.real_DI)

    def backward_G(self):
        # Static Seg loss
        b, c, h, w = self.real_SS.size() #b, 8, h, w

        # BCE Seg loss
        self.loss_G = self.criterionBCE(
            torch.log_softmax(self.fake_SS, dim=1),
            self.real_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, h, w)
        )
        self.loss_G.backward()

    def optmize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()  # set gradients to zero
        self.backward_G()  # calculate graidents
        self.optimizer_G.step()  # udpate weights

    def get_current_loss(self):
        loss = dict()
        loss['G'] = self.loss_G.item()
        return loss

    def onehot2rgb(self, output):
        device = output.device
        train2color = {0: (128, 128, 128), 1: (128, 0, 0), 2: (64, 64, 128), 3: (0, 0, 0), 4: (0, 128, 64),
                       5: (128, 64, 128), 6: (0, 0, 192), 7: (192, 192, 0)}
        # output = F.log_softmax(output)
        c, h, w = output.size()
        b = 1
        output = output.view(1, c, h, w)
        pred = output.permute(0, 2, 3, 1).view(-1, self.num_classes).max(1)[1].view(b, 1, h, w)
        pred_color = torch.zeros(b, 3, h, w)
        for k, v in train2color.items():
            pred_r = torch.zeros(b, 1, h, w)
            pred_r[pred == k] = v[0]
            pred_g = torch.zeros(b, 1, h, w)
            pred_g[pred == k] = v[1]
            pred_b = torch.zeros(b, 1, h, w)
            pred_b[pred == k] = v[2]
            pred_color += torch.cat((pred_r, pred_g, pred_b), 1)
        return (pred_color.to(device) / 255. - 0.5) * 2

    def get_current_visuals(self):
        visuals = dict()
        visuals['real_DI'] = self.real_DI[0]
        visuals['real_SI'] = self.real_SI[0]
        visuals['real_SS'] = self.onehot2rgb(self.real_SS[0])[0]
        visuals['fake_SS'] = self.onehot2rgb(self.fake_SS[0])[0]
        return visuals

    def get_statistic_errors(self):
        errors = dict()
        errors['S_PA'], errors['S_MA'], errors['S_MIOU'], errors['S_FWIOU'] = self.iou(self.real_SS[0], self.fake_SS[0])
        return errors

    def test(self, samples):
        self.set_input(samples)
        self.forward()
        visuals = self.get_current_visuals()
        error = self.get_statistic_errors()
        return visuals, error

    def iou(self, target, pred):
        # input
        # target,pred onehot B,C,H,W

        # pred numpy HXW
        # target numpy HXW
        # nclasses = 8

        ## step 1 onehot2map
        c, h, w = target.size()
        b = 1
        target = target.view(b, c, h, w)
        target = target.permute(0, 2, 3, 1).view(-1, self.num_classes).max(1)[1].view(h, w)

        pred = pred.view(b, c, h, w)
        pred = pred.permute(0, 2, 3, 1).view(-1, self.num_classes).max(1)[1].view(h, w)

        Metrics = SegMetrics(target.cpu().numpy(), pred.cpu().numpy(), self.num_classes)
        if self.isTrain:
            return Metrics.mean_metric()
        else:
            return Metrics.perclass_metric()

