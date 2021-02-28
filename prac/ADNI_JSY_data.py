import pandas as pd
import setting as st
import utils
import numpy as np

""" tadpole dataset """
smri_info_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/smri_orig_info.csv'
data = pd.read_csv(smri_info_dir)

n_img = (data['Image ID']).unique().shape[0] # (21280, )

""" label """
label_type = (data['Research Group']).unique() # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']

""" phase """
phase_type = data['Phase'].unique() # [ANDI 1, ADNI GO, ADNI 2, ADNI 3, nan]
n_phase_wo_nan = data.count()['Phase'] # (21274)
data['Phase'].value_counts() # 9317+9105+1653+1199


""" Check the distribution depending on the Phase and Label """
n_img_label_wise=[[] for _ in range(phase_type.shape[0])]
# ADNI 1 : [[2694, 1596, 4815, 0, 0, 0, 0],
# ADNI GO : [179, 0, 237, 773, 10, 0, 0],
# ADNI 2 : [2748, 1043, 622, 2605, 0, 1701, 598],
# ADNI 3 : [777, 107, 350, 210, 4, 90, 115],
# NaN : [4, 0, 0, 2, 0, 0, 0]]
count_n_BL = 0
for i_label in label_type :
    for j, j_phase in enumerate(phase_type):
        if j_phase in phase_type: # if the 'phase' is in the [1, 2, go, 3]
            n_img_label_wise[j].append(((data['Research Group'] == i_label) & (data['Phase'][:] == j_phase)).sum())
        else: # if the 'phase' is not in the [1, 2, go ,3]
            count_n_BL+=1
            n_img_label_wise[j].append(((data['Research Group'] == i_label) & (data['Phase'][:].isnull())).sum())


""" if phase is nan, the phase is ADNI 2. """
for tmp_i in range(data['Phase'][:].isnull().shape[0]):
    if data['Phase'][tmp_i] not in phase_type:
        data['Phase'][tmp_i] = 'ADNI 2'


""" sbject """
n_sbj = (data['Subject ID']).unique().shape[0] # (2428, )
sbj_type = data['Subject ID'].unique()

""" baseline ONLY """
n_img_label_wise_BL = [np.zeros(label_type.shape) for _ in range(phase_type.shape[0])] # (phase, label)
last_name = ''
cur_name = ''
cur_phase = ''
last_phase = ''
count_n_BL = 0
count_if_both_appear = 0
for i in range(n_img):
    cur_name = data['Subject ID'][i] # subject id
    cur_phase = data['Phase'][i]
    if last_name == cur_name:
        if cur_phase != last_phase and (data['Phase'][i] == 'ADNI 1' or data['Phase'][i] == 'ADNI 2') and (data['Research Group'][i] == 'CN' or data['Research Group'][i] == 'AD'):
            count_if_both_appear +=1
        last_phase = cur_phase

        pass
    else:
        count_n_BL += 1
        n_img_label_wise_BL[phase_type.tolist().index(data['Phase'][i])][label_type.tolist().index(data['Research Group'][i])] += 1
        last_name = cur_name
        last_phase = cur_phase



""" description """
uniq_desc = data['Description'].unique().shape[0] # 164
count = 0
for tmp_i in range (uniq_desc):
    if 'REP' in np.char.upper(data['Description'].unique()[tmp_i]).tolist():
        # print(np.char.upper(data['Description'].unique()[tmp_i]))
        if 'REPE' not in np.char.upper(data['Description'].unique()[tmp_i]).tolist():
            print(np.char.upper(data['Description'].unique()[tmp_i]))
            count += 1
print(count)



""" count baseline wo MPRAGE REPEAT """
n_img_label_wise_BL = [np.zeros(label_type.shape) for _ in range(phase_type.shape[0])]  # (phase, label)
last_name = ''
cur_name = ''
cur_phase = ''
last_phase = ''
count_n_BL = 0
count_if_both_appear = 0
for i in range(n_img):
    ## TODO : REPE should not be in the Description
    if 'REPE' not in np.char.upper(data['Description'][i]).tolist():
        cur_name = data['Subject ID'][i]  # subject id
        cur_phase = data['Phase'][i]
        if last_name == cur_name:
            if cur_phase != last_phase and (cur_phase == 'ADNI 1' and last_phase == 'ADNI 2'):
                count_if_both_appear += 1
            last_phase = cur_phase

            pass
        else:
            count_n_BL += 1
            n_img_label_wise_BL[phase_type.tolist().index(data['Phase'][i])][
                label_type.tolist().index(data['Research Group'][i])] += 1
            last_name = cur_name
            last_phase = cur_phase


""" Check whether there is any subject whose baseline is ADNI 2, and captured in ADNI 1 after all. """
n_img_label_wise_BL = [np.zeros(label_type.shape) for _ in range(phase_type.shape[0])]  # (phase, label)
last_name = ''
cur_name = ''
cur_phase = ''
last_phase = ''
count_n_BL = 0
count_if_both_appear = 0
# ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']
# ADNI 1 : [[231, 200, 414, 0, 0, 0, 0],
# ADNI GO : [0, 0, 0, 142, 5, 0, 0],
# ADNI 2 : [202, 159, 0, 191, 0, 178, 111],
# ADNI 3 : [332, 66, 193, 0, 4, 0, 0],
# NaN : [0, 0, 0, 0, 0, 0, 0]]
for i in range(n_img):
    if 'REPE' not in np.char.upper(data['Description'][i]).tolist():
        cur_name = data['Subject ID'][i]  # subject id
        cur_phase = data['Phase'][i]
        if last_name == cur_name:
            pass
        else:
            count_n_BL += 1
            if cur_phase != 'ADNI 2':
                n_img_label_wise_BL[phase_type.tolist().index(data['Phase'][i])][
                    label_type.tolist().index(data['Research Group'][i])] += 1
                last_name = cur_name
            else:
                count_next = 0
                flag_both_appear = False
                while(1):
                    count_next +=1
                    next_name = data['Subject ID'][i+ count_next]
                    next_phase = data['Phase'][i + count_next]

                    if cur_name != next_name:
                        break
                    else:
                        # print(next_phase)
                        if next_phase == 'ADNI 1':
                            flag_both_appear = True

                if flag_both_appear:
                    pass
                else:
                    n_img_label_wise_BL[phase_type.tolist().index(data['Phase'][i])][
                        label_type.tolist().index(data['Research Group'][i])] += 1
                    last_name = cur_name

print('sMCI pMCI labeling')
"""  sMCI pMCI label """
"""
1. fill out the DXCURREN
2. Mask 'reversion' and 's/pMCI'
"""
dir_DXSUM = '/DataCommon/chpark/ADNI_orig_JSY/info/DXSUM_PDXCONV_ADNIALL.csv'
data_dxsum = pd.read_csv(dir_DXSUM)
conv_list = ['DXCHANGE', 'DXCURREN', 'DXCONV']

