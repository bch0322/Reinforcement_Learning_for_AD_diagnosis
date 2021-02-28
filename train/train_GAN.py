import os
import sys
import utils as ut
from torch.autograd import Variable
import torch
import torch.nn as nn
import setting as st
import setting_2 as fst
import numpy as np
from modules import *
from plot import *

def train(config, fold, model_1, model_2, dict_loader, optimizer_1, optimizer_2, scheduler_1, scheduler_2, hyperParam, list_dir_save_model, list_dir_heatmap, dir_pyplot, Validation=True, Test_flag = True):

    train_loader = dict_loader['train']
    val_loader = dict_loader['val']
    test_loader = dict_loader['test']


    EMS = ut.eval_metric_storage()
    list_selected_EMS = []
    list_ES = []
    for i_tmp in range(len(st.list_standard_eval_dir)):
        list_selected_EMS.append(ut.eval_selected_metirc_storage())
        list_ES.append(ut.EarlyStopping(delta=0, patience=hyperParam.early_stopping_patience, verbose=True))

    print('training')
    optimizer_1.zero_grad()
    optimizer_1.step()
    optimizer_2.zero_grad()
    optimizer_2.step()

    """ epoch """
    fake_buffer = ReplayBuffer()
    for epoch in range(1, config.num_epochs+1):
        scheduler_1.step()
        scheduler_2.step()
        print(" ")
        print("---------------  epoch {} ----------------".format(epoch))

        """ print learning rate """
        for param_group in optimizer_1.param_groups:
            print('current LR : {}'.format(param_group['lr']))

        """ train """
        model_1, model_2, optimizer_1, optimizer_2, EMS, fake_buffer = ut.train_classification_model_VAGAN(config, fold, epoch, EMS, train_loader,
                                                                                                           model_1, model_2, optimizer_1, optimizer_2, fake_buffer, hyperParam)

        """ val """
        if Validation == True:
            print("------------------  val  --------------------------")
            dict_result = ut.eval_classification_model_VAGAN(config, fold, epoch, val_loader, model_1, hyperParam, list_dir_heatmap[0])
            ut.MRI_plot_fake_img(config, fold, epoch, val_loader, model_1, hyperParam, list_dir_heatmap[0])
            val_loss = dict_result['Loss']
            acc = dict_result['Acc']
            auc = dict_result['AUC']

            print('Fold : %d, Epoch [%d/%d] val Loss = %f val Acc = %f' % (fold, epoch, config.num_epochs, val_loss, acc))

            """ save the metric """
            EMS.dict_val_metric['val_loss'].append(val_loss)
            EMS.dict_val_metric['val_acc'].append(acc)
            EMS.dict_val_metric['val_auc'].append(auc)
            EMS.val_step.append(EMS.total_train_step)

            n_stacking_loss_for_selection = 5
            if len(EMS.dict_val_metric['val_loss_queue']) > n_stacking_loss_for_selection:
                EMS.dict_val_metric['val_loss_queue'].popleft()
            EMS.dict_val_metric['val_loss_queue'].append(val_loss)
            EMS.dict_val_metric['val_mean_loss'].append(np.mean(EMS.dict_val_metric['val_loss_queue']))

            """ save model """
            for i_tmp in range(len(list_selected_EMS)):
                save_flag = ut.model_save_through_validation(fold, epoch, EMS=EMS,
                                                             start_eval_epoch=hyperParam.early_stopping_start_epoch,
                                                             selected_EMS=list_selected_EMS[i_tmp],
                                                             ES=list_ES[i_tmp],
                                                             model=model_1,
                                                             dir_save_model=list_dir_save_model[i_tmp],
                                                             metric_1=st.list_standard_eval[i_tmp], metric_2='',
                                                             save_flag=False)

        if Test_flag== True:
            print("------------------  test _ test dataset  --------------------------")
            dict_result = ut.eval_classification_model_VAGAN(config, fold, epoch, test_loader, model_1, hyperParam, list_dir_heatmap[0])
            ut.MRI_plot_fake_img(config, fold, epoch, test_loader, model_1, hyperParam, list_dir_heatmap[0])

            acc = dict_result['Acc']
            test_loss = dict_result['Loss']

            """ pyplot """
            EMS.test_acc.append(acc)
            EMS.test_loss.append(test_loss)
            EMS.test_step.append(EMS.total_train_step)

            print('number of test samples : {}'.format(len(test_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] test Loss = %f test Acc = %f' % (fold, epoch, config.num_epochs, test_loss, acc))


        """ learning rate decay"""
        EMS.LR.append(optimizer_1.param_groups[0]['lr'])
        # scheduler.step()
        # scheduler.step(val_loss)

        """ plot the chat """
        if epoch % 1 == 0:
            ut.plot_training_info_1(fold, dir_pyplot, EMS,  hyperParam, flag='percentile', flag_match=False)

        ##TODO : early stop only if all of metric has been stopped
        tmp_count = 0
        for i in range(len(list_ES)):
            if list_ES[i].early_stop == True:
                tmp_count += 1
        if tmp_count == len(list_ES):
            break

    """ release the model """
    del model_1, EMS
    torch.cuda.empty_cache()
