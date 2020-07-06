# -*- coding: utf-8 -*-
import os 
from collections import OrderedDict
import shutil

def main():
    # Intialization
    OUR_METHOD = 'FAWDN+'
    save_dir = 'FAWDN_SAVE'
    #os.removedirs(save_dir)

    methods = OrderedDict()
    record_mean = False

    # Transfer to OrderedDict() for the future use
    with open('/home/ser606/clh/eval_for_SR/PSNR_SSIM_Results_BI_model.txt') as f:
        for line in f:
            if 'Method' in line:
                name = line.split()[-3][:-1]
                bm_name = line.split()[-1]
                if name not in methods.keys():
                    methods[name] = OrderedDict()
                if bm_name not in methods[name].keys():
                    methods[name][bm_name] = OrderedDict()
                    methods[name][bm_name]['img_name'] = list()
                    methods[name][bm_name]['PSNR'] = list()
                    methods[name][bm_name]['SSIM'] = list()
                    # methods[name][bm_name]['IFC'] = list   # TODO
            elif '--' in line:
                record_mean = True
            elif '**' in line: continue
            elif record_mean:
                split_line = line.split()
                methods[name][bm_name]['avg_PSNR'] = float(split_line[-3])
                methods[name][bm_name]['avg_SSIM'] = float(split_line[-1])
                record_mean = False
            else:
                split_line = line.split()
                methods[name][bm_name]['img_name'].append(split_line[-5][:-1])
                methods[name][bm_name]['PSNR'].append(float(split_line[-3]))
                methods[name][bm_name]['SSIM'].append(float(split_line[-1]))

    bm_list = methods[OUR_METHOD].keys()
    for bm_name in bm_list:
        print('Processing %s'%bm_name)
        save_bm_dir = os.path.join(save_dir, bm_name)
        if not os.path.exists(save_bm_dir):
            os.makedirs(save_bm_dir)
        our_results = methods[OUR_METHOD][bm_name]['PSNR']
        comp_total = [True for _ in range(len(our_results))]

        # Comparision (You can edit the comparision criterion here)
        for name in methods.keys():
            if name == OUR_METHOD: continue
            comp_method = list(map(lambda x,y:x > y, our_results, methods[name][bm_name]['PSNR']))
            comp_total = list(map(lambda x,y:x and y, comp_total, comp_method))
        comp_idx = [idx for idx, item in enumerate(comp_total) if item]

        # save files
        for name in methods.keys():
            if name == 'OurA+': continue  # TODO: A+
            for sel_idx in comp_idx:
                psnr = methods[name][bm_name]['PSNR'][sel_idx]
                old_img_name = methods[name][bm_name]['img_name'][sel_idx]
                new_img_name = os.path.splitext(old_img_name)[0] + '_%f'%psnr + os.path.splitext(old_img_name)[-1]
                src_path = os.path.join('SR', '', name, bm_name, 'x3', old_img_name) # TODO: adaptive scale
                shutil.copy(src_path, os.path.join(save_bm_dir, new_img_name))


if __name__ == '__main__':
    main()