# from __future__ import division
from modules import *


class network(nn.Module):
    def __init__(self, config):
        """ init """

        self.cur_shape = np.array([st.x_size, st.y_size, st.z_size])
        # self.kernel = [3, 3, 3, 3, 3]
        # self.strides = [1, 2, 2, 2, 1]
        # self.padding = [0, 0, 0, 0, 0]

        self.kernel = [3]
        self.strides = [1]
        self.padding = [0]
        super(network, self).__init__()

        in_p = 8
        f_out = [8, 16, 32, 64, 128]
        self.widening_factor = 2
        self.inplanes = in_p * self.widening_factor
        f_out = [f_out[i] * self.widening_factor for i in range(len(f_out))]
        self.cur_shape = np.array([st.x_size, st.y_size, st.z_size])
        super(network, self).__init__()

        """ low-level """
        norm = 'in'
        act_func = 'relu'
        self.layer1 = BasicConv_Block(in_planes=1, out_planes=self.inplanes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, act_func=act_func, norm_layer=norm, bias=False)

        self.layer2 = Bottleneck(inplanes=self.inplanes, planes=f_out[0], kernel_size=3, stride=2)
        self.inplanes = f_out[0] * Bottleneck.expansion

        self.layer3 = Bottleneck(inplanes=self.inplanes, planes=f_out[1], kernel_size=3, stride=2)
        self.inplanes = f_out[1] * Bottleneck.expansion

        self.layer4 = Bottleneck(inplanes=self.inplanes, planes=f_out[2], kernel_size=3, stride=2)
        self.inplanes = f_out[2] * Bottleneck.expansion

        self.layer5 = Bottleneck(inplanes=self.inplanes, planes=f_out[3], kernel_size=3, stride=2)
        self.inplanes = f_out[3] * Bottleneck.expansion

        self.layer6 = Bottleneck(inplanes=self.inplanes, planes=f_out[4], kernel_size=3, stride=1)
        self.inplanes = f_out[4] * Bottleneck.expansion

        self.classifier_aux_1 = nn.Conv3d(self.inplanes, st.num_class, kernel_size=1, stride=1, bias=True)

        self.inplanes = f_out[2] * Bottleneck.expansion

        # placeholder for the gradients
        self.gradients = []

        """ input dependent LRLC"""
        self.LRLC_1 = Input_Dependent_LRLC(in_channels = self.inplanes, out_channels=self.inplanes, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1, act_func=act_func, norm_layer=None, bias=False, n_K = 4)

        n_head = 1
        flag_norm = False
        self.ISAB_1 = ISAB(dim_in=self.inplanes, dim_out=self.inplanes, num_heads=n_head , num_inds=64, ln=flag_norm)
        self.ISAB_2 = ISAB(dim_in=self.inplanes, dim_out=self.inplanes, num_heads=n_head , num_inds=64, ln=flag_norm)
        self.PMA = PMA(dim=self.inplanes, num_heads=n_head , num_seeds=32, ln=flag_norm)
        self.SAB_1 = SAB(dim_in=self.inplanes, dim_out=self.inplanes, num_heads=n_head , ln=flag_norm)
        self.SAB_2 = SAB(dim_in=self.inplanes, dim_out=self.inplanes, num_heads=n_head , ln=flag_norm)
        self.MAB_1 = MAB(dim_Q=f_out[4] * Bottleneck.expansion, dim_KV=self.inplanes, dim_out=self.inplanes, num_heads=4, ln=flag_norm)

        self.inplanes = f_out[2] * Bottleneck.expansion
        self.classifier = nn.Linear(self.inplanes, st.num_class, bias=True)

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients.append(grad)

    def forward(self, x_0, flag_1, *args):
        """ encoder """
        # (B, 1, 194, 233, 184)
        if flag_1 == 0:
            x_context = x_0
            x_context = self.layer1(x_context)
            x_context = self.layer2(x_context)
            x_context = self.layer3(x_context)
            x_context_a = self.layer4(x_context)
            x_context_b = self.layer5(x_context_a)
            x_context_b = self.layer6(x_context_b)  ## (b, 256, 9, 11, 9)

            feature = [x_context_a, x_context_b, x_context_a]

            dict_result = {
                "logits": None,  # batch, 2
                "Aux_logits": [None],  # batch, 2
                "logitMap": None,  # batch, 2, w, h ,d
                "l1_norm": None,
                "final_evidence": None,  # batch, 2, w, h, d
                "featureMaps": feature,
            }
            return dict_result

        elif flag_1 == 1:
            """ context information """
            x_context = x_0[1] ## (b, 512, 19, 23, 19)

            x_context_2 = nn.AvgPool3d(kernel_size=x_context.size()[-3:], stride=1)(x_context)
            out_aux_1 = self.classifier_aux_1(x_context_2)
            out_aux_1 = out_aux_1.view(out_aux_1.size(0), -1)

            """ subtle info """
            x_subtle = x_0[2]
            x_subtle = self.LRLC_1(x_subtle)
            action_a = args[0][0].detach()  # (B, 1, 19, 23, 19)

            """ Q : context, KV : subtle """
            list_batch = []
            list_batch_2 = []
            for i_batch in range(x_subtle.size(0)):
                index = (action_a[i_batch].view(-1) == 1).nonzero().squeeze(1)

                tmp_kv_x = torch.index_select(x_subtle[i_batch].view(x_subtle[i_batch].size(0), -1), -1, index).unsqueeze(0)
                tmp_q_x = x_context[i_batch].unsqueeze(0)
                tmp_q_x = tmp_q_x.view(tmp_q_x.size(0), tmp_q_x.size(1), -1)
                # tmp_q_x = tmp_q_x.mean(dim=-1, keepdim=True)

                tmp_x = torch.mean(tmp_kv_x, dim=-1)
                list_batch_2.append(tmp_x)

                # after selection
                tmp_kv_x = self.ISAB_1(tmp_kv_x)
                tmp_kv_x = self.ISAB_2(tmp_kv_x)
                tmp_kv_x = self.PMA(tmp_kv_x)
                tmp_kv_x = self.SAB_1(tmp_kv_x)
                tmp_kv_x = self.SAB_2(tmp_kv_x)

                # after query
                tmp_kv_x = self.MAB_1(tmp_q_x, tmp_kv_x)

                tmp_kv_x = torch.mean(tmp_kv_x, dim=-1)
                list_batch.append(tmp_kv_x.squeeze(-1))

            out_0 = torch.cat(list_batch, dim=0)
            out_0 = self.classifier(out_0)
            out_0 = out_0.view(out_0.size(0), -1)

            dict_result = {
                "logits": out_0,  # batch, 2
                "Aux_logits": [out_aux_1],  # batch, 2
                "logitMap": None,  # batch, 2, w, h ,d
                # "logitMap": [class_evidence_aux_1, class_evidence_aux_2],  # batch, 2, w, h ,d
                "l1_norm": None,
                "final_evidence": None,  # batch, 2, w, h, d
                "featureMaps": None,
            }
            return dict_result

def Model(config):
    model = network(config)
    return model