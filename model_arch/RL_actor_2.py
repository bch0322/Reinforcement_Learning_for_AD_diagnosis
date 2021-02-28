# from __future__ import division
from modules import *

class network(nn.Module):
    def __init__(self, config):
        """ init """
        super(network, self).__init__()

        in_p = 8
        f_out = [8, 16, 32, 64, 128]
        self.widening_factor = 2
        self.inplanes = in_p * self.widening_factor
        f_out = [f_out[i] * self.widening_factor for i in range(len(f_out))]


        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.n_classes = config.num_classes
        norm = 'in'
        act_func = 'relu'

        self.outplanes = 256
        self.inplanes = f_out[2] * Bottleneck.expansion
        self.sq_local = nn.Sequential(
            nn.Conv3d(self.inplanes, self.outplanes, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(self.outplanes, affine=True),
            nn.ReLU(inplace=True),
        )

        self.inplanes = f_out[4] * Bottleneck.expansion
        self.sq_global = nn.Sequential(
            nn.Conv3d(self.inplanes, self.outplanes, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(self.outplanes, affine=True),
            nn.ReLU(inplace=True),
        )

        self.inplanes = self.outplanes
        self.LRLC_1 = Input_Dependent_LRLC(in_channels = self.inplanes, out_channels=self.inplanes, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1, act_func=act_func, norm_layer=norm, bias=False, n_K = 4)
        self.actor = nn.Sequential(
            nn.Conv3d(self.inplanes, self.inplanes//2, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm3d(self.inplanes//2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inplanes//2, 2, kernel_size=1, stride=1, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_0, *args):
        """ input """
        x_local = x_0[0].detach()
        x_global = x_0[1].detach()

        """actor"""
        x_local = self.sq_local(x_local)
        out_0 = self.LRLC_1(x_local)
        action_logit_a = self.actor(out_0)  # (B, 512, 19, 23, 19)

        # return [action_logit_a], [state_values_a]
        return [action_logit_a], [None]

def Model(config):
    model = network(config)
    return model