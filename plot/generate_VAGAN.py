import numpy as np
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import torch
import torch.nn as nn
import utils as ut

def get_heatmap_2class(config, fold, epoch, loader, model, hyperParam, dir_heatmap=None):
    """ free all GPU memory """
    torch.cuda.empty_cache()

    """ loss """
    criterion = nn.L1Loss()

    """ load the fold list for test """
    list_test_data = DL.concat_class_of_interest(config, fold, list_class=st.list_class_for_test,
                                                 flag_tr_val_te='test')
    test_loader = DL.convert_Dloader_3(config.v_batch_size, list_test_data[0], list_test_data[1], list_test_data[2],
                                       list_test_data[3], is_training=False, num_workers=0, shuffle=False)


    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    if model_dir !=None:
        model.load_state_dict(torch.load(model_dir))
    model.eval()

    """ param for accuracy """
    test_batchnum = 0

    """ eval """
    patch_size = st.patch_size
    stride_between_patches = st.patch_stride
    count = 0



    with torch.no_grad():
        for datas, labels, alabel, mlabel in test_loader:
            datas_backup=datas
            count += 1

            """ prediction """
            datas_pred = Variable(datas).cuda()
            dict_result = model(datas_pred)
            output_logit = dict_result['logits']
            prob = nn.Softmax(dim=1)(output_logit)
            pred = prob.argmax(dim=1, keepdim=True)  # batch, 1


            """ padding """
            logit_map = np.zeros_like(datas)
            attn_1_map = np.zeros_like(datas)
            attn_2_map = np.zeros_like(datas)
            final_evidence_map = np.zeros_like(datas)

            """ padding """
            m = nn.ConstantPad3d(patch_size//2, 0)
            datas = m(datas)

            """ loop as much as the size of strides """
            for i in range(stride_between_patches):
                print("i : {0}".format(i))
                for j in range(stride_between_patches):
                    # print("j : {0}".format(j))
                    for k in range(stride_between_patches):

                        """ input"""
                        data = Variable(datas[:, :, i:, j:, k:]).cuda()
                        labels = Variable(labels.long()).cuda()

                        """ run classification model """
                        dict_result = model(data)
                        logitMap = dict_result['logitMap']
                        attn_1 = dict_result['attn_1']
                        attn_2 = dict_result['attn_2']
                        final_evidence = dict_result['final_evidence']


                        shape = logitMap.shape
                        for a in range(shape[-3]):
                            for b in range(shape[-2]):
                                for c in range(shape[-1]):
                                    if logitMap is not None:
                                        logit_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            logitMap[:, pred[0].data.cpu().numpy()[0], a, b, c].data.cpu().numpy()

                                    if attn_1 is not None:
                                        attn_1_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            attn_1[:, 0, a, b, c].data.cpu().numpy()
                                    if attn_2 is not None:
                                        attn_2_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            attn_2[:, 0, a, b, c].data.cpu().numpy()
                                    if final_evidence is not None:
                                        final_evidence_map[:, 0, a * stride_between_patches + i, b * stride_between_patches + j, c * stride_between_patches + k] = \
                                            final_evidence[:, pred[0].data.cpu().numpy()[0], a, b, c].data.cpu().numpy()
                            # logit_list.append(test_output_1.cpu().numpy())

            torch.cuda.empty_cache()
            print("finished a sample!")
            for sample in range(logit_map.shape[0]):
                tmp_save_dir = dir_heatmap + '/fold_{0}'.format(fold,

                                                                                )
                ut.make_dir(dir=tmp_save_dir, flag_rm=False)
                ut.save_featureMap_tensor(datas_backup[sample][0], tmp_save_dir, 'input_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))

                if logitMap is not None:
                    ut.save_featureMap_numpy(logit_map[sample][0], tmp_save_dir, 'logit_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = logit_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/1_logit_map_{}'.format(count), fig_title='Original Logit Map', thresh=0.2, percentile=1)

                if attn_1 is not None:
                    ut.save_featureMap_numpy(attn_1_map[sample][0], tmp_save_dir, 'attn_1_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = attn_1_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/2_attn_map_1_{}'.format(count), fig_title='Attention Map 1', thresh=0.2, percentile=1)

                if attn_2 is not None:
                    ut.save_featureMap_numpy(attn_2_map[sample][0], tmp_save_dir, 'attn_2_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = attn_2_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/2_attn_map_2_{}'.format(count), fig_title='Attention Map 2', thresh=0.2, percentile=1)

                if final_evidence is not None:
                    ut.save_featureMap_numpy(final_evidence_map[sample][0], tmp_save_dir, 'final_evidence_map_{0}_gt_{1}_pred_{2}'.format(
                count,
                st.list_selected_for_test[labels[sample].data.cpu().numpy()],
                st.list_selected_for_test[pred[sample].data.cpu().numpy()[0]]
                ))
                    orig_img = datas_backup[sample][0]
                    heatmap_img = final_evidence_map[sample][0]
                    ut.plot_heatmap_with_overlay(orig_img=orig_img, heatmap_img=heatmap_img, save_dir=tmp_save_dir + '/3_final_evidence_{}'.format(count), fig_title='Final Evidence', thresh=0.2, percentile=1)
