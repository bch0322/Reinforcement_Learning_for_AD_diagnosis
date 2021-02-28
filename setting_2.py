""" flags """
""" preprocessing for the data """
flag_orig_npy = False
flag_fold_index = False
flag_print_trainAcc = False


""" loss stage 1"""
flag_stage_1 = False
flag_stage_1_RL = True

flag_stage1_loss_1 = True
flag_stage1_loss_2 = False
flag_stage1_loss_3 = True

# """ loss stage 2"""
# flag_stage_2 = False
# flag_stage2_loss_1 = True # cls
# flag_stage2_loss_2 = True # norm
# flag_stage2_loss_3 = True # GAN
# flag_stage2_loss_4 = True # cyc
# flag_stage2_loss_5 = True # ide
# flag_stage2_loss_6 = False # KL
# flag_stage2_loss_7 = True # dis


"""
0 : always      
1 : no cropping       
2 : (1:1) cropping
"""
flag_MC_dropout = False
flag_translation = False
flag_translation_subject_wise = True
flag_translation_ratio = 0
flag_translation_ratio_2 = False
flag_eval_translation = False

flag_cropping = True
flag_cropping_subject_wise = True
flag_random_flip = True

flag_Avgpool = False
flag_Gaussian_blur = False

flag_scheduler_cosine = True
flag_scheduler_expo = False

flag_plot_CAM = False
flag_plot_mean_DAM = False

flag_pretrained = False


# pre, fine
flag_copy_including = True
dict_pretrained_layer = {
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4',
    'layer5': 'layer5',
    'layer6': 'layer6',
    'classifier_aux_1' : 'classifier_aux_1'

                         }
flag_freeze_including = False
list_freeze_layer = ['layer1',
                     'layer2',
                     'layer3',
                     'layer4',
                     'layer5',
                     'layer6',
                     'classifier_aux_1'
                     ]
