import setting as st

""" model """

from model_arch import RL_pred_2
from model_arch import RL_actor_2


def construct_model(config, flag_model_num = 0):
    """ construct model """
    if flag_model_num == 0:
        model_num = st.model_num_0
    elif flag_model_num == 1:
        model_num = st.model_num_1



    if model_num == 0:
        pass
    elif model_num == 92:
        model = RL_pred_2.Model(config).cuda()
    elif model_num == 93:
        model = RL_actor_2.Model(config).cuda()
    return model



