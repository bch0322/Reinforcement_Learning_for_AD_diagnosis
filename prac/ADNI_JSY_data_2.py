import pandas as pd
import setting as st
import utils
import numpy as np

smri_info_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/smri_orig_info.csv'
data = pd.read_csv(smri_info_dir)
data = data.sort_values(by=['RID', 'Image ID'])

""" info """
n_img = (data['Image ID']).unique().shape[0]  # (21280, )
label_type = (data['Research Group']).unique()  # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']

""" if phase is nan, the phase is ADNI 2. """
phase_type = data['Phase'].unique()  # [ANDI 1, ADNI GO, ADNI 2, ADNI 3, nan]
n_phase_wo_nan = data.count()['Phase']  # (21274)
data['Phase'].value_counts()  # 9317+9105+1653+1199
for tmp_i in range(data['Phase'][:].isnull().shape[0]):
    if data['Phase'][tmp_i] not in phase_type:
        data['Phase'][tmp_i] = 'ADNI 2'

""" sbject """
n_sbj = (data['Subject ID']).unique().shape[0]  # (2428, )
sbj_type = data['Subject ID'].unique()

""" visit 1 """
visit_type = (data['Visit 1']).unique()  # []
count = 0
for tmp_i in range(data['Visit 1'].unique().shape[0]):
    if 'ADNI2' in data['Visit 1'].unique()[tmp_i]:
        count += 1

""" file diagnosis change """
diag_change_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/diagnosis_change.csv'
diag_change_data = pd.read_csv(diag_change_dir)
diag_change_data = diag_change_data.sort_values(by=['RID', 'ImageID'])
potential_pMCI = diag_change_data['ImageID'].unique()

""" file DXSUM """
DXSUM_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/DXSUM_PDXCONV_ADNIALL.csv'
DXSUM_data = pd.read_csv(DXSUM_dir)
DXSUM_data = DXSUM_data.sort_values(by=['RID', 'EXAMDATE'])

""" start """
""" start """
""" start """
flag_stndard_MCI = True  # When False, the standard would be applied
list_standard_MCI = ['m36', 'm48', 'm60', 'm72', 'm84', 'm96']
list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36', 'm48', 'm60', 'm72', 'm84', 'm96']
# list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36']
list_standard_pMCI = ['m06', 'm12', 'm18', 'm24', 'm36']

## TODO : Without considering MPRAGE Repeat

# ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']
# ADNI 1 : [[231, 200, 414, 0, 0, 0, 0],
# ADNI GO : [0, 0, 0, 142, 5, 0, 0],
# ADNI 2 : [202, 159, 0, 191, 0, 178, 111],
# ADNI 3 : [332, 66, 193, 0, 4, 0, 0],
# NaN : [0, 0, 0, 0, 0, 0, 0]]

num_img_label_wise_BL = [np.zeros(label_type.shape) for _ in range(phase_type.shape[0])]  # (phase, label)
ID_img_label_wise_BL = [[[] for _ in range(label_type.shape[0])] for _ in
                        range(phase_type.shape[0])]  # (phase, label)
phase_label_wise_imageID = [[[] for _ in range(4)] for _ in range(2)]
last_PTID = ''
cur_PTID = ''

cur_phase = ''
last_phase = ''
count_n_BL = 0

count_Rev_from_AD = np.zeros(2)
count_empty_in_diag = np.zeros(2)
count_not_matching = np.zeros(2)
count_if_both_appear = 0

list_test = []
list_test_2 = []
excluded_year = ['2019', '2020']
for i in range(n_img):
    cur_image_ID = data['Image ID'][i]
    cur_PTID = data['Subject ID'][i]  # subject id
    cur_RID = data['RID'][i]
    cur_phase = data['Phase'][i]
    cur_research_group = data['Research Group'][i]
    cur_visit = data['Visit 1'][i]
    cur_StudyDate = data['Study Date'][i]
    cur_ImgProt = data['Imaging Protocol'][i]
    if last_PTID == cur_PTID:  # same subject
        pass

    else:
        if not any(s in cur_StudyDate for s in excluded_year):
            if (cur_phase == 'ADNI 1') and (cur_visit == 'ADNI Screening') and (cur_ImgProt[cur_ImgProt.index(
                    'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                'Field Strength') + 4] == '1.5'):
                # if (cur_phase == 'ADNI 1') and (cur_visit == 'ADNI Baseline' or cur_visit == 'ADNI Screening'):
                # if cur_visit == 'ADNI Baseline':
                count_n_BL += 1

                ID_img_label_wise_BL[
                    phase_type.tolist().index(cur_phase)][
                    label_type.tolist().index(cur_research_group)
                ].append(cur_image_ID)

                num_img_label_wise_BL[
                    phase_type.tolist().index(cur_phase)][
                    label_type.tolist().index(cur_research_group)
                ] += 1
                last_PTID = cur_PTID

                if cur_research_group == 'CN':
                    phase_label_wise_imageID[0][0].append(cur_image_ID)

                elif cur_research_group == 'AD':
                    phase_label_wise_imageID[0][1].append(cur_image_ID)

                elif 'MCI' in cur_research_group:  ##TODO : MCI conversion
                    tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)]
                    if tmp_df['DXCURREN'].empty:
                        # print('empty : {}'.format(cur_PTID))
                        count_empty_in_diag[0] += 1
                        pass
                    elif tmp_df['DXCURREN'].iloc[0] != 2:
                        # print('not matching : {}'.format(cur_PTID))
                        count_not_matching[0] += 1
                        pass
                    else:
                        ## TODO #1 no reversion
                        check_rev = False  # default : False
                        for i_tmp in range(tmp_df.shape[0]):
                            if i_tmp == 0:
                                prev_DX = tmp_df['DXCURREN'].iloc[i_tmp]
                            else:
                                if prev_DX == 3 and prev_DX > tmp_df['DXCURREN'].iloc[i_tmp]:
                                    # print('REV : {}'.format(cur_PTID))
                                    count_Rev_from_AD[0] += 1
                                    check_rev = True

                        ## TODO #2 At least one diagnosis is more than 36
                        check_more_than_36 = flag_stndard_MCI  # default : False
                        for i_tmp in range(tmp_df.shape[0]):
                            if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                check_more_than_36 = True

                        ##
                        if check_rev == False and check_more_than_36 == True:
                            tmp_df_2 = tmp_df[tmp_df['DXCURREN'] == 3]  # take only AD
                            if tmp_df_2['VISCODE2'].empty:
                                # pass
                                phase_label_wise_imageID[0][2].append(cur_image_ID)
                            elif not any(
                                    tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                phase_label_wise_imageID[0][2].append(cur_image_ID)
                            elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                phase_label_wise_imageID[0][3].append(cur_image_ID)
                            else:  ##TODO : else
                                pass
                                # MCI_Conversion[0][0].append(cur_image_ID)
                                # print('else')


            # elif (cur_phase == 'ADNI 2') and ('Year' not in cur_visit):
            # elif (cur_phase == 'ADNI 2') and ('ADNI2 Initial' in cur_visit or 'ADNI2 Screening' in cur_visit):
            elif (cur_phase == 'ADNI 2') and ('ADNI2 Screening' in cur_visit) and (cur_ImgProt[cur_ImgProt.index(
                    'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                'Field Strength') + 4] == '3.0'):

                ## TODO : check if appear in 'ADNI 1'
                tmp_df = data[(data['Subject ID'] == cur_PTID)]
                tmp_df = tmp_df[(tmp_df['Phase'] == 'ADNI 1')]
                if tmp_df.shape[0] != 0:
                    count_if_both_appear += 1
                count_n_BL += 1

                ID_img_label_wise_BL[
                    phase_type.tolist().index(cur_phase)][
                    label_type.tolist().index(cur_research_group)
                ].append(cur_image_ID)

                num_img_label_wise_BL[
                    phase_type.tolist().index(cur_phase)][
                    label_type.tolist().index(cur_research_group)
                ] += 1
                last_PTID = cur_PTID

                if cur_research_group == 'CN':
                    phase_label_wise_imageID[1][0].append(cur_image_ID)

                elif cur_research_group == 'AD':
                    phase_label_wise_imageID[1][1].append(cur_image_ID)

                elif 'EMCI' in cur_research_group:
                    tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)]
                    if tmp_df['DXCHANGE'].empty:
                        # print('empty : {}'.format(cur_PTID))
                        count_empty_in_diag[1] += 1
                    elif tmp_df['DXCHANGE'].iloc[0] != 2:
                        # print('not matching : {}'.format(cur_PTID))
                        count_not_matching[1] += 1
                    else:
                        ## TODO #1 no reversion
                        check_rev = False  # default : False
                        for i_tmp in range(tmp_df.shape[0]):
                            if tmp_df['DXCHANGE'].iloc[i_tmp] == 8 or tmp_df['DXCHANGE'].iloc[i_tmp] == 9:
                                count_Rev_from_AD[1] += 1
                                print(cur_PTID)
                                check_rev = True

                        ## TODO #2 At least one diagnosis is more than 36
                        check_more_than_36 = flag_stndard_MCI  # default : False
                        for i_tmp in range(tmp_df.shape[0]):
                            if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                check_more_than_36 = True

                        if check_rev == False and check_more_than_36 == True:
                            tmp_df_2 = tmp_df[(tmp_df['DXCHANGE'] == 3) | (tmp_df['DXCHANGE'] == 5)]  # take only AD
                            if tmp_df_2['VISCODE2'].empty:
                                # pass
                                phase_label_wise_imageID[1][2].append(cur_image_ID)
                            elif not any(
                                    tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                phase_label_wise_imageID[1][2].append(cur_image_ID)
                            elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                phase_label_wise_imageID[1][3].append(cur_image_ID)
                            else:  ##TODO : else
                                pass
                                # MCI_Conversion[0][0].append(cur_image_ID)
                                # print('else')

            else:  # not ADNI 1 or 2
                pass
# print(num_img_label_wise_BL)

print("count_appear_in_both : {}".format(count_if_both_appear))
print("ADNI1, NC : {}".format(len(phase_label_wise_imageID[0][0])))
print("ADNI1, AD : {}".format(len(phase_label_wise_imageID[0][1])))
print("ADNI2, NC : {}".format(len(phase_label_wise_imageID[1][0])))
print("ADNI2, AD : {}".format(len(phase_label_wise_imageID[1][1])))
print('------------------------------------------------------')
print("ADNI1, sMCI : {}".format(len(phase_label_wise_imageID[0][2])))
print("ADNI1, pMCI : {}".format(len(phase_label_wise_imageID[0][3])))
print("ADNI2, sMCI : {}".format(len(phase_label_wise_imageID[1][2])))
print("ADNI2, pMCI : {}".format(len(phase_label_wise_imageID[1][3])))

print('empty : {}'.format(count_empty_in_diag))
print('not matching : {}'.format(count_not_matching))
print('Rev from AD : {}'.format(count_Rev_from_AD))
