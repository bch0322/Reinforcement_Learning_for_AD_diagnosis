import socket
from datetime import datetime
import os
import GPUtil
import setting_2 as fst
import shutil
import utils as ut
import numpy as np
import time

"""GPU connection"""
devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
# devices = '6'
# os.environ["CUDA_VISIBLE_DEVICES"] = devices

dir_etc_info = './etc_info'
ut.make_dir(dir_etc_info)
f = open(dir_etc_info + '/gpu_num.txt', 'w')
tmp = '{}_{}'.format(socket.gethostname(), devices)
print(tmp)
f.write(tmp)
f.close()

""" data """
data_type_num = 2  ## 0, 5
list_data_type = ['Density',
                  'ADNI_JSY',                   # 1
                  'ADNI_JSY_between_1_and_2',   # 2
                  'ADNI_Jacob_64',              # 3
                  'ADNI_Jacob_256',             # 4
                  'ADNI_AAL_256',               # 5
                  'ADNI_AAL_256_2',             # 6
                  'ADNI_AAL_256_3',             # 7
                  'ADNI_AAL_256_4'              # 8
                  ]


""" data normalization """
data_norm_type_num = 1
list_data_norm_type = ['woNorm', 'minmax', 'gaussian']


""" task selection """
if 'ADNI_AAL_256' in list_data_type[data_type_num]:
    list_class_type = ['NC', 'MCI', 'sMCI', 'pMCI', 'AD']
    list_class_for_train = [1, 0, 0, 0, 1]
    list_class_for_test = [1, 0, 0, 0, 1]
    # list_class_for_train = [0, 0, 1, 1, 0] #
    # list_class_for_test = [0, 0, 1, 1, 0] #

    # list_class_for_train = [1, 0, 1, 1, 1]  #
    # list_class_for_test = [1, 0, 0, 0, 1]  #
    # list_class_for_train = [1, 0, 1, 1, 1]  #
    # list_class_for_test = [0, 0, 1, 1, 0]  #

    list_class_for_total = [1, 1, 1, 1, 1] # for plotting

elif list_data_type[data_type_num] == 'ADNI_JSY':
    list_class_type = ['NC', 'MCI', 'AD']
    list_class_for_train = [1, 0, 1]
    list_class_for_test = [1, 0, 1]

    # list_class_for_train = [1, 1, 0]
    # list_class_for_test = [1, 1, 0]

    list_class_for_total = [1, 1, 1]  # for plotting

elif 'ADNI_JSY_between_1_and_2' in list_data_type[data_type_num]:
    list_class_type = ['NC', 'AD', 'sMCI', 'pMCI']

    list_class_for_train = [1, 1, 0, 0]
    list_class_for_test = [1, 1, 0, 0]

    # list_class_for_train = [0, 0, 1, 1]
    # list_class_for_test = [0, 0, 1, 1]

    list_eval_type = ['1_to_2', '2_to_1']
    num_eval_choise = 0

    list_class_for_total = [1, 1, 1, 1]  # for plotting


elif 'Density' in list_data_type[data_type_num]:
    list_class_type = ['NC', 'MCI', 'sMCI', 'pMCI', 'AD']
    list_class_for_train = [1, 0, 0, 0, 1]
    list_class_for_test = [1, 0, 0, 0, 1]
    list_class_for_total = [1, 1, 1, 1, 1] # for plotting



""" selected task """
list_selected_for_train = [] # ['NC', 'MCI', 'AD']
list_selected_for_test = [] # ['NC', 'MCI', 'AD']
list_selected_for_total = [] # ['NC', 'MCI', 'AD']

for i in range(len(list_class_for_total)):
    if list_class_for_total[i] == 1:
        list_selected_for_total.append(list_class_type[i])
    if list_class_for_train[i] == 1:
        list_selected_for_train.append(list_class_type[i])
    if list_class_for_test[i] == 1:
        list_selected_for_test.append(list_class_type[i])
num_class = len(list_selected_for_train)

""" eval metric """
# list_standard_eval_dir = ['/val_loss', '/val_acc', '/val_auc']
list_standard_eval_dir = ['/val_loss']
list_standard_eval = ['{}'.format(list_standard_eval_dir[i][1:]) for i in range(len(list_standard_eval_dir))]
list_eval_metric = ['MAE', 'RMSE', 'R_squared',  'Acc', 'Sen', 'Spe', 'AUC']

""" parmas """
kfold = 10
start_fold = 1
end_fold = 10

## TODO : hyperParam
max_num_loss = 20
class hyperParam_storage_1():
    def __init__(self):
        super(hyperParam_storage_1, self).__init__()
        self.name = 'stage_1'
        self.epoch = 200
        self.batch_size = 2
        self.iter_to_update = 1
        self.v_batch_size = self.batch_size

        self.lr = 1e-4
        self.LR_decay_rate = 0.98
        self.step_size = 1

        self.early_stopping_start_epoch = 1
        self.early_stopping_patience = 50
        self.weight_decay = 1e-5
        self.smoothing_label = 0.1

        self.loss_type = ['cls', 'norm', 'aux_cls']
        self.flag_aux_cls = [0, 0, 1]

        self.name_aux_cls = ['_65']
        self.smoothing_label_aux = [0.1]
        self.num_aux_cls = len(self.name_aux_cls)
        self.num_total_loss = len(self.flag_aux_cls) - sum(self.flag_aux_cls) + len(self.name_aux_cls)

        self.loss_lambda = {
            self.loss_type[0]: 1.0,  # cls
            self.loss_type[1]: 0.01,  # norm
            self.loss_type[2] + self.name_aux_cls[0]: 1.0,  # aux_cls_1
            # self.loss_type[2] + self.name_aux_cls[1]: 1.0,  # aux_cls_2
            # self.loss_type[2] + self.name_aux_cls[2]: 1.0,  # aux_cls_3
            # self.loss_type[2] + self.name_aux_cls[2]: 1.0,  # aux_cls_3
        }



class hyperParam_storage_2():
    def __init__(self):
        super(hyperParam_storage_2, self).__init__()
        self.name = 'stage_2'
        self.epoch = 200
        # self.batch_size = 1
        # self.iter_to_update = 4
        # self.v_batch_size = self.batch_size

        self.lr = 1e-4
        self.LR_decay_rate = 0.98
        self.step_size = 1

        # self.early_stopping_start_epoch = self.epoch
        # self.early_stopping_patience = self.epoch
        self.weight_decay = 1e-5
        # self.loss_type = ['cls', 'norm', 'GAN', 'cyc', 'ide', 'gen_dif_L1', 'dis']
        # self.loss_lambda = {
        #     self.loss_type[0]: 1.0,  # cls
        #     self.loss_type[1]: 100.0,  # norm
        #     self.loss_type[2]: 1.0,  # GAN
        #     self.loss_type[3]: 10.0,  # cyc (l1)
        #     self.loss_type[4]: 5.0,  # ide (l1)
        #     self.loss_type[5]: 0.0,  # gen_dif_L1
        #     self.loss_type[6]: 0.5,  # discri
        # }
        self.smoothing_label = 0.1

hyperParam_s1 = hyperParam_storage_1()
hyperParam_s2 = hyperParam_storage_2()

""" model setting """
# dir_preTrain_1 = '/home/chpark/pretrained/200921_FCN68_aux32/1_MIL_base/val_loss'
dir_preTrain_1 = '/DataCommon/chpark/pretrained/200927/2_BagNet/4_65_LS/1_MIL_base/val_loss'
model_arch_dir = "/model_arch"

model_num_0 = 92  # to train classifier, generator
model_num_1 = 93  # to train discriminator

model_name = [None] * 100

model_name[92] = "RL_pred_2"
model_name[93] = "RL_actor_2"

dir_to_save_1 = './1_' + model_name[model_num_0]
dir_to_save_2 = './2_' + model_name[model_num_1]

""" directory """
# dir_root = '/Data/chpark'
# dir_root = '/DataCommon/chpark/ADNI'
dir_root = '/home/chpark/ADNI'

data_size = None
num_modality = 1
orig_data_dir = '/DataCommon/chpark/ADNI_orig_JSY'
tmp_data_path = '/' + list_data_type[data_type_num]
exp_data_dir = dir_root + '/ADNI_exp' + tmp_data_path
tadpole_dir = dir_root + '/TADPOLE-challenge/TADPOLE_D1_D2.csv'

if list_data_type[data_type_num] == 'Density':
    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    x_range = [0, 121]
    y_range = [0, 145]
    z_range = [0, 121]
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]

elif list_data_type[data_type_num] == 'ADNI_JSY':
    x_range = [0, 193]
    y_range = [0, 229]
    z_range = [0, 193]
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]

elif list_data_type[data_type_num] == 'ADNI_JSY_between_1_and_2':
    x_range = [0, 193]
    y_range = [0, 229]
    z_range = [0, 193]
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]


elif list_data_type[data_type_num] == 'ADNI_AAL_256_4':

    template_x_range = [58, 201+1]
    template_y_range = [38, 220+1]
    template_z_range = [28, 161+1]
    spare_value = 25
    dir_list = ['/NORMAL', '/MCI', '/AD', '/sMCI', '/pMCI']
    x_range = template_x_range
    y_range = template_y_range
    z_range = template_z_range

    x_range[0] -= spare_value
    x_range[1] += spare_value

    y_range[0] -= spare_value
    y_range[1] += spare_value

    z_range[0] -= spare_value
    z_range[1] += spare_value

    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_range[1] - z_range[0]


""" 1. raw npy dir """
orig_npy_dir = exp_data_dir + '/orig_npy'
ADNI_fold_image_path = []
ADNI_fold_age_path = []
ADNI_fold_MMSE_path = []

ADNI_fold_image_path_2 = []
ADNI_fold_age_path_2 = []
ADNI_fold_MMSE_path_2 = []

if list_data_type[data_type_num] == 'ADNI_JSY_between_1_and_2' :
    for i in range(len(list_class_type)):
        ADNI_fold_image_path.append(orig_npy_dir + "/ADNI1_" + str(list_class_type[i]) + "_image.npy")
        ADNI_fold_age_path.append(orig_npy_dir + "/ADNI1_" + str(list_class_type[i]) + "_age.npy")
        ADNI_fold_MMSE_path.append(orig_npy_dir + "/ADNI1_" + str(list_class_type[i]) + "_MMSE.npy")

    for i in range(len(list_class_type)):
        ADNI_fold_image_path_2.append(orig_npy_dir + "/ADNI2_" + str(list_class_type[i]) + "_image.npy")
        ADNI_fold_age_path_2.append(orig_npy_dir + "/ADNI2_" + str(list_class_type[i]) + "_age.npy")
        ADNI_fold_MMSE_path_2.append(orig_npy_dir + "/ADNI2_" + str(list_class_type[i]) + "_MMSE.npy")
else:
    for i in range(len(list_class_type)):
        ADNI_fold_image_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_image.npy")
        ADNI_fold_age_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_age.npy")
        ADNI_fold_MMSE_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_MMSE.npy")

""" 2. fold index """
fold_index_dir = exp_data_dir + '/fold_index'
if list_data_type[data_type_num] == 'ADNI_JSY_between_1_and_2' :
    train_index_dir = []
    val_index_dir = []
    test_index_dir = []

    train_index_dir_2 = []
    val_index_dir_2 = []
    test_index_dir_2 = []
    for i in range(len(list_class_type)):
        train_index_dir.append(fold_index_dir + '/1_train_index_' + list_class_type[i])
        val_index_dir.append(fold_index_dir + '/1_val_index_' + list_class_type[i])
        test_index_dir.append(fold_index_dir + '/1_test_index_' + list_class_type[i])
    for i in range(len(list_class_type)):
        train_index_dir_2.append(fold_index_dir + '/2_train_index_' + list_class_type[i])
        val_index_dir_2.append(fold_index_dir + '/2_val_index_' + list_class_type[i])
        test_index_dir_2.append(fold_index_dir + '/2_test_index_' + list_class_type[i])

else:
    train_index_dir = []
    val_index_dir = []
    test_index_dir = []
    for i in range(len(list_class_type)):
        train_index_dir.append(fold_index_dir + '/train_index_' + list_class_type[i])
        val_index_dir.append(fold_index_dir + '/val_index_' + list_class_type[i])
        test_index_dir.append(fold_index_dir + '/test_index_' + list_class_type[i])

""" gaussian """
if len(list_selected_for_train) == 2 :
    tmp_dir = './gaussian'
    gaussian_dir = tmp_dir + '/' + '{}_{}'.format(list_selected_for_train[0], list_selected_for_train[1])

""" t_test """
if len(list_selected_for_train) == 2 :
    tmp_dir = './t_test'
    ttest_dir = tmp_dir + '/' + '{}_{}'.format(list_selected_for_train[0], list_selected_for_train[1])
heatmap_dir = './heatmap' + '/' + '{}_{}'.format(list_selected_for_train[0], list_selected_for_train[1])

""" experiment description """
exp_date = str(datetime.today().year) + '%02d'%datetime.today().month + '%02d'% datetime.today().day
# exp_name = '/Exp_1'
exp_title = '/tmp'
exp_description = "classification"

""" print out setting """
print(socket.gethostname())
print("data : {}".format(exp_date))

print(' ')
print("Dataset for train : {}".format(list_selected_for_train))
print("model arch 1 : {}".format(model_name[model_num_0]))

if fst.flag_stage_1_RL == True:
    print("model arch 2 : {}".format(model_name[model_num_1]))

print(' ')
print("data type : {}".format(list_data_type[data_type_num]))


size_translation = 8


# 193, 229, 193
# crop_size = 185
# max_crop_size = [crop_size, crop_size, crop_size]
# min_crop_size = [int(crop_size), int(crop_size), int(crop_size)]

max_crop_size = [185, 221, 185]
min_crop_size = max_crop_size

crop_pad_size = (0, 0, 0, 0, 0, 0)
# crop_pad_size = (3, 3, 3, 3, 3, 3)

""" data size"""
data_size = [1, x_size, y_size, z_size]

# if fst.flag_cropping == True :
#     if fst.flag_Avgpool == False :
#         data_size = [1, max_crop_size[0], max_crop_size[1], max_crop_size[2]]
#     elif fst.flag_Avgpool == True :
#         data_size = [1, max_crop_size[0] // 2, max_crop_size[1] // 2, max_crop_size[2] //2]
# else:
#     if fst.flag_Avgpool == True :
#         data_size = [1, x_size // 2, y_size // 2, z_size //2]
#     else:
#         data_size = [1, x_size, y_size, z_size]


"""openpyxl setting """
push_start_row = 2
