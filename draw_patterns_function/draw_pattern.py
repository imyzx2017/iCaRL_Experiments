import matplotlib.pyplot as plt
import h5py
import os
import numpy as np
from scipy.misc import imresize

def sigmoid(x, derivative=False):
    sigm = 2. / (1. + np.exp(-1*x)) - 1
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def load_snd_mask(txt_path, task_num, repeat_num):
    file = open(txt_path, 'r')
    if task_num == 10: n_codes = 15
    elif task_num == 20: n_codes = 20
    elif task_num == 100: n_codes = 50
    mask = np.zeros((task_num, n_codes))
    cid = 0
    for line in file.readlines():
        code_string = line.split('code:')[-1].strip()
        code_string = code_string.split(';')[:-1]
        code_string = [float(x) for x in code_string]
        mask[cid] = code_string
        cid+=1
    mask = mask.reshape(1, -1)
    final_mask = np.repeat(mask, repeat_num, axis=0)
    return final_mask


METHODS = ['SNN']
# METHODS = ['ANN Single', 'EWC', 'iCaRL', 'GEM', 'ANN Joint Train', 'SNN']
# task_num = 100
root_dir = 'results/patterns'
# method = 'iCaRL'
# fv_idx = 1
# nosc = 'NOSC50-5-2'
for task_num in [10]:
    if task_num == 10:
        dataset = 'MNIST'
        dpi = 500
        scale = 1
        neuron_num = 50
        sample_num = 50
        repeat_num = 10
    elif task_num == 20:
        dataset = 'EMNIST20'
        dpi = 500
        scale = 1
        neuron_num = 67
        sample_num = 50
        repeat_num = 10
    elif task_num == 100:
        dataset = 'CIFAR100'
        dpi = 1000
        scale = 10
        neuron_num = 167
        sample_num = 10
        repeat_num = 3
    else:
        raise Exception('unsupport task number!')
    for method in METHODS:
        save_dir = os.path.join(root_dir, dataset, method.replace(' ', ''))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for fv_idx in [1]:
            hdf5_search_path = os.path.join(root_dir, dataset, method.replace(' ', ''), 'FV%d-stability'%fv_idx)
            if method == 'iCaRL':
                root_dir=''
                hdf5_search_path = os.path.join(root_dir, 'results/patterns/iCaRL', dataset, 'nodropout', 'hidden%d_1Layers'%neuron_num, 'FV%d-stability'%fv_idx)
            elif method == 'SNN':
                data_dir = os.path.join(root_dir, 'data', 'SNN', dataset)
                for dir in os.listdir(data_dir):
                    if 'FV%d' % fv_idx in dir:
                        hdf5_search_path = os.path.join(data_dir, dir)
                        subdir = os.listdir(hdf5_search_path)[0]
                        hdf5_search_path = os.path.join(hdf5_search_path, subdir)#, 'TestsetStability')
                # hdf5_search_path = os.path.join('SNN-pattern', '%s-SNN-GC'%dataset, 'FV%d'%fv_idx, 'stability')

            for file in os.listdir(hdf5_search_path):
                if 'Neurons_Firing_rate_during_CL.hdf5' in file:
                    if dataset in file:
                        dst_file = file
                        dst_path = os.path.join(hdf5_search_path, file)

            matrix = h5py.File(dst_path, 'r')['data']
            if method == 'SNN':
                # plt.title('%s %s GC FV%d Neural Population Response Graph' % (dataset, method, fv_idx), fontsize=8)
                plt.title('MNIST10 SNN15-30-15(ANOSC15-3-1 MaxMin) NEW FV1 Train NPRG', fontsize=8)
            else:
                if method == 'ANN Joint Train':
                    method_name = 'ANN(Offline)'
                elif method == 'ANN Single':
                    method_name = 'ANN(Base)'
                else: method_name=method
                plt.title('%s %s New FV%d Neural Population Response Graph(after sigmoid)'%(dataset, method_name, fv_idx), fontsize=8)
            # Plotting Snd_Mask

            color_value_max = matrix.value.max()
            if method == 'SNN':
                im = plt.imshow(matrix, cmap='jet', vmax=color_value_max)
                # snd_mask_path = os.path.join(root_dir, 'data\SNN/NOSC')
                # for file in os.listdir(snd_mask_path):
                #     if 'SndMask%d_' % task_num in file:
                #         snd_mask_path = os.path.join(snd_mask_path, file)
                #         snd_mask = load_snd_mask(snd_mask_path, task_num, repeat_num)
                # snd_mask = imresize(snd_mask.transpose(), (matrix.shape[0], repeat_num)) / 255.0
                # new_mat = np.zeros((matrix.shape[0], matrix.shape[1]+repeat_num))
                # new_mat[:, repeat_num:] = matrix
                # for i in range(task_num):
                #     new_mat[(i*neuron_num):(i*neuron_num+neuron_num), (i*sample_num):(i*sample_num+repeat_num)] = \
                #         snd_mask[(i*neuron_num):(i*neuron_num+neuron_num),:]*10.0
                # im = plt.imshow(new_mat, cmap='jet', vmax=20.0)
            else:
                matrix = sigmoid(matrix.value)
                im = plt.imshow(matrix, cmap='jet', vmax=1.0)

            plt.colorbar(im, shrink=0.5)
            neuron_num = matrix.shape[0] / task_num
            y_ticks = np.arange(int(neuron_num/2)-1, matrix.shape[0]+1 , neuron_num*scale)
            temp = (scale*np.arange(int(task_num/scale))).tolist()
            plt.yticks(y_ticks, (scale*np.arange(int(task_num/scale))).tolist())
            sample_num = matrix.shape[1] / task_num
            x_ticks = np.arange(int(sample_num/2)-1+10, matrix.shape[1]+1 , sample_num*scale)
            plt.xticks(x_ticks, (scale*np.arange(int(task_num/scale))).tolist())
            plt.ylabel('Class Index')
            plt.xlabel('Learning Step')
            plt.gca().set_aspect(matrix.shape[1]/matrix.shape[0])
            plt.tight_layout()

            save_name = method.replace(' ', '') + '_'+ dst_file.split('.')[0]
            save_path = os.path.join(save_dir, save_name)
            print(save_path)
            plt.savefig(save_path, dpi=dpi)
            # plt.show()
            plt.close('all')
