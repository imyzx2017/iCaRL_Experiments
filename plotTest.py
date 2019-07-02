# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure()
# plt.title('123')
# plt.xticks([])
# plt.yticks([])
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=0.94, bottom=0.01,left=0.01,right=0.99,hspace=0,wspace=0)
# cmap = plt.get_cmap('gist_ncar')
# color_list = np.linspace(0, 1, 100)
# current_color = cmap(color_list[0])
# plt.show()


import os
import matplotlib.pyplot as plt
import numpy as np
import math


def draw_robust_result(dir_path, fv_index, task_num=10):
    plt.figure(figsize=(20, 10))
    for each_dir in os.listdir(dir_path):
        temp_path = dir_path + each_dir + '/'
        noise_name = each_dir.split('-')[-1]
        if noise_name == 'NOISE(0)':
            noise_final_name = 'TestSet-ORIGINAL'
        else:
            noise_final_name = 'TestSet-{}'.format(noise_name)
        for files in os.listdir(temp_path):
            if '.csv' in files and 'FV{}'.format(fv_index) in files:
                if 'withoutdropout' in files:
                    temp_name = 'nodropout'
                    file_path = temp_path + files
                    data = np.loadtxt(file_path)

                    t = range(1, task_num + 1)
                    plt.plot(t, data, label='iCaRL-MNIST-MLP(15-50-10)-{}-{}'.format(noise_final_name, temp_name))
                    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99, hspace=0, wspace=0)
                    plt.ylabel('Average accuracy')
                    plt.xlabel('Number of tasks')
                    plt.ylim([0, 1.1])
                    plt.hold

                else:
                    temp_name = 'dropout(0.5)'
    plt.legend()
    plt.title('MNIST-FV{}-Robust_Test-TestsetAddingNoise-Result'.format(fv_index))
    plt.savefig('MNIST_FV{}_ROBUST_TEST_RESULT.png'.format(fv_index), dpi=300)
    plt.show()
    # plt.close()

def draw_changeFV_result(dir_path, fv_index, task_num=10):
    plt.figure(figsize=(20, 10))
    for each_dir in os.listdir(dir_path):
        temp_path = dir_path + each_dir + '/'
        FV_name = each_dir.split('-')[-1]
        if FV_name == 'New':
            FV_final_name = 'FV{}-New'.format(fv_index)
        else:
            FV_final_name = 'FV{}-ORIGINAL'.format(fv_index)
        for files in os.listdir(temp_path):
            if '.csv' in files and 'FV{}'.format(fv_index) in files:
                if 'withoutdropout' in files:
                    temp_name = 'nodropout'
                    file_path = temp_path + files
                    data = np.loadtxt(file_path)

                    t = range(1, task_num + 1)
                    plt.plot(t, data, label='iCaRL-MNIST-MLP(15-50-10)-{}-{}'.format(FV_final_name, temp_name))
                    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99, hspace=0, wspace=0)
                    plt.ylabel('Average accuracy')
                    plt.xlabel('Number of tasks')
                    plt.ylim([0, 1.1])
                    plt.hold

                else:
                    temp_name = 'dropout(0.5)'
    plt.legend()
    plt.title('MNIST-FV{}-ChangeFV-Result'.format(fv_index))
    plt.savefig('MNIST-FV{}-ChangeFV-Result.png'.format(fv_index), dpi=300)
    plt.show()
    # plt.close()

def sigmoid(input_x, beta=1.0):
    return 1.0 / (1.0 + math.exp(-beta * input_x))

def inverse_sigmoid(input_x):
    return (-math.log((1.0 - input_x)/input_x))

def inverse_sigmoid_in_FV(dirs, fv_index, task_num=10, beta=1.0):
    if task_num == 10:
        dataset = 'MNIST-FV'
        input_dim = 15
        samples = 50
        orig_hidden_dim = 50
        SNN_STRUCTURE='15-50-10'
    elif task_num == 20:
        dataset = 'EMNIST-FV'
        input_dim = 20
        samples = 50
        orig_hidden_dim = 67
        SNN_STRUCTURE = '20-67-20'
    elif task_num == 100:
        dataset = 'CIFAR100-FV'
        input_dim = 50
        samples = 10
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'

    # trainset_list = ['FV-train50test50-BeforeSigmoid']
    trainset_list = ['0619_OutliersDetection_data']
    # trainset_list = ['FV-train50test50']  # for MNIST, EMNIST
    # trainset_list = ['FV-train10test10']  # for CIFAR100
    # trainset_list = ['FV-train10test10-BeforeSigmoid']  # for CIFAR100
    # save_dir = dirs + 'FVnew-train10test10/'
    # save_path = save_dir + dataset + '/'

    # beta = calculate_new_parameter_for_sigmoid(task_num)
    # beta = 1.0
    for traintest_dir in os.listdir(dirs):
        for traintest in trainset_list:
            if traintest==traintest_dir and not '.rar' in traintest_dir:
                dataset_dirs = dirs + traintest_dir + '/'
                for dataset_dir in os.listdir(dataset_dirs):
                    if dataset_dir == dataset:
                        files_path = dataset_dirs + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'FV{}'.format(fv_index) in files and not '.naf' in files:
                                f = open(files_path + files)
                                data = f.readlines()
                                f.close()


                                if 'trainset' in files:
                                    set_traindata_list = []
                                    samples_num = 0
                                    activation_num = 0
                                    diff_classes_activation_trainset_metric = np.zeros(shape=(task_num, samples, input_dim))
                                    set_name = 'trainset'
                                    for idx in range(task_num):
                                        for item in data:
                                            if item.split(',')[0] == 'class:{}'.format(idx):
                                                if samples_num==samples:
                                                    samples_num=0

                                                data_str = item.split('code:')[-1]
                                                for each_data in data_str.split(';'):
                                                    if each_data == '\n':
                                                        activation_num=0
                                                        samples_num+=1
                                                    else:
                                                        diff_classes_activation_trainset_metric[idx, samples_num, activation_num] = float(each_data)
                                                        activation_num+=1
                                                        set_traindata_list.append(float(each_data))
                                                        # set_traindata_list.append(sigmoid(float(each_data), beta))
                                elif 'testset' in files:
                                    set_testdata_list = []
                                    samples_num = 0
                                    activation_num = 0
                                    diff_classes_activation_testset_metric = np.zeros(
                                        shape=(task_num, samples, input_dim))
                                    set_name = 'testset'
                                    for idx in range(task_num):
                                        for item in data:
                                            if item.split(',')[0] == 'class:{}'.format(idx):
                                                if samples_num==samples:
                                                    samples_num=0
                                                data_str = item.split('code:')[-1]
                                                for each_data in data_str.split(';'):
                                                    if each_data == '\n':
                                                        activation_num=0
                                                        samples_num+=1
                                                    else:
                                                        diff_classes_activation_testset_metric[idx, samples_num, activation_num] = sigmoid(float(each_data))
                                                        activation_num+=1
                                                        set_testdata_list.append(float(each_data))
                                                        # set_testdata_list.append(sigmoid(float(each_data), beta))
    return set_traindata_list, set_testdata_list, diff_classes_activation_trainset_metric, diff_classes_activation_testset_metric

def Remake_newFV(dirs, task_num=100, fv_index=1,x=1.0):
    if task_num == 10:
        dataset = 'MNIST-FV'
    elif task_num == 20:
        dataset = 'EMNIST-FV'
    elif task_num == 100:
        dataset = 'CIFAR100-FV'

    beta = calculate_new_parameter_for_sigmoid(task_num, x=x)

    # trainset_list = ['FV-train50test50-BeforeSigmoid']
    trainset_list = ['0619_OutliersDetection_data']# for CIFAR100
    save_dir = dirs + 'FVnew-train50test50-0619/'  # for CIFAR100

    # trainset_list = ['FV-train10test10-BeforeSigmoid']  # for CIFAR100
    # save_dir = dirs + 'FVnew-train10test10/' # for CIFAR100
    save_path = save_dir + dataset + '/'

    for traintest_dir in os.listdir(dirs):
        for traintest in trainset_list:
            if traintest == traintest_dir and not '.rar' in traintest_dir:
                dataset_dirs = dirs + traintest_dir + '/'
                for dataset_dir in os.listdir(dataset_dirs):
                    if dataset_dir == dataset:
                        files_path = dataset_dirs + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'FV{}'.format(fv_index) in files and not '.naf' in files:
                                f = open(files_path + files)
                                data = f.readlines()
                                f.close()

                                if 'beforeSigmoid' in files:
                                    name = files.split('(beforeSigmoid)')[0] + '(AfterNewSigmoid)' + files.split('(beforeSigmoid)')[-1]
                                f_save = open(save_path + name, 'w')
                                for item in data:
                                    inital_str = item.split('code:')[0] + 'code:'
                                    data_str = item.split('code:')[-1]
                                    f_save.write(inital_str)
                                    for each_data in data_str.split(';'):
                                        if each_data == '\n':
                                            f_save.write(each_data)
                                        else:
                                            f_save.write('{};'.format(sigmoid(float(each_data), beta)))
                                f_save.close()

def calculate_new_parameter_for_sigmoid(task_num=10, x=20.0):
    if not task_num==100:
        beta = math.log(19) / x      # default /20.0
        return beta
    else:
        beta = math.log(19) / 100.0     # default /100.0
        return beta

def Draw_Neuron_Activation_0611(class_num=7, binary=False, train='train', path='D:\\Projects\\Projects\\pytorch_Projects\\iCaRL-TheanoLasagne\\data\\0611_data\\'):
    if binary:
        metric = np.zeros((15, 50))
        if train=='train':
            f = open(path + 'new_binary_FV_trainset.txt', 'r+')
            data = f.readlines()
            f.close()
        else:
            train = 'test'
            f = open(path + 'new_binary_FV_testset.txt', 'r+')
            data = f.readlines()
            f.close()
        idx = 0
        for item in data:
            current_class = int(item.split(',')[0].split(':')[-1])
            if current_class == class_num:
                data = np.array(item.split(':')[-1].split(';')[:-1])
                for i, activation in enumerate(data):
                    metric[i, idx] = float(activation)
                idx += 1
    else:
        metric = np.zeros((15, 50))
        if train=='train':
            f = open(path + 'new_FV_trainset.txt', 'r+')
            data = f.readlines()
            f.close()
        else:
            train='test'
            f = open(path + 'new_FV_testset.txt', 'r+')
            data = f.readlines()
            f.close()
        idx = 0
        for item in data:
            current_class = int(item.split(',')[0].split(':')[-1])
            if current_class == class_num:
                data = np.array(item.split(':')[-1].split(';')[:-1])
                for i, activation in enumerate(data):
                    metric[i, idx] = float(activation)
                idx += 1

    h = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(top=0.97, bottom=0.03, left=0.04, right=0.96, hspace=0, wspace=0)
    plt.imshow(metric, cmap='jet')
    if binary:
        plt.title('Binary_Fv({}set) samples_activations from class {}'.format(train, class_num))
    else:
        plt.title('Fv({}set) samples_activations from class {}'.format(train, class_num))
    plt.xlabel('Samples')
    plt.ylabel('Activations')
    position = h.add_axes([0.97,0.15,0.01,0.685])#位置[左,下,右,上]
    plt.colorbar(cax=position)#orientation='horizontal')
    if binary:
        plt.savefig('Binary_Fv_{}set_Activation_from_class{}.png'.format(train, class_num), dpi=300)
    else:
        plt.savefig('Fv_{}set_Activation_from_class{}.png'.format(train, class_num), dpi=300)
    plt.show()

def Draw_Neuron_ALL_Activation_0611(task_num=10, binary=False, train='train', path='D:\\Projects\\Projects\\pytorch_Projects\\iCaRL-TheanoLasagne\\data\\0611_data\\'):
    if binary:
        metric = np.zeros(16*task_num, 50)
        if train=='train':
            f = open(path + 'new_binary_FV_trainset.txt', 'r+')
            data = f.readlines()
            f.close()
        else:
            train = 'test'
            f = open(path + 'new_binary_FV_testset.txt', 'r+')
            data = f.readlines()
            f.close()

        for class_n in range(task_num):
            idx = 0
            for item in data:
                current_class = int(item.split(',')[0].split(':')[-1])
                if current_class == class_n:
                    current_data = np.array(item.split(':')[-1].split(';')[:-1])
                    for i, activation in enumerate(current_data):
                        metric[i+16*class_n, idx] = float(activation)
                    idx += 1
    else:
        metric = np.zeros((16*task_num, 50))
        if train=='train':
            f = open(path + 'epoch4_trainset_fc_pattern_FV1.txt', 'r+')
            # f = open(path + 'new_FV_trainset.txt', 'r+')
            data = f.readlines()
            f.close()
        else:
            train='test'
            f = open(path + 'epoch4_testset_fc_pattern_FV1.txt', 'r+')
            # f = open(path + 'new_FV_testset.txt', 'r+')
            data = f.readlines()
            f.close()
        idx = 0
        for class_n in range(task_num):
            idx = 0
            for item in data:
                current_class = int(item.split(',')[0].split(':')[-1])
                if current_class == class_n:
                    current_data = np.array(item.split(':')[-1].split(';')[:-1])
                    for i, activation in enumerate(current_data):
                        metric[i + 16 * class_n, idx] = float(activation)
                    idx += 1

    h = plt.figure(figsize=(100, 10))
    plt.subplots_adjust(top=0.97, bottom=0.03, left=0.04, right=0.96, hspace=0, wspace=0)
    plt.imshow(metric, cmap='jet', aspect='auto')
    if binary:
        plt.title('Binary_Fv({}set) samples_activations from Allclass {}'.format(train))
    else:
        plt.title('Fv({}set) samples_activations from Allclass'.format(train))
    plt.xlabel('Samples')
    plt.ylabel('Activations')
    # plt.xticks(range(1, 50, 50))
    # plt.yticks(range(0, 9))

    # plt.margins(1, 0.01)
    position = h.add_axes([0.97,0.15,0.01,0.685])#位置[左,下,右,上]
    plt.colorbar(cax=position)#orientation='horizontal')
    if binary:
        plt.savefig('Binary_Fv_{}set_Activation_from_Allclass.png'.format(train), dpi=300)
    else:
        plt.savefig(path + 'Fv_{}set_Activation_from_Allclass.png'.format(train), dpi=300)
    plt.show()



if __name__ == '__main__':

    # Draw_Neuron_Activation_0611(class_num=7, binary=False, train='test')
    # Draw_Neuron_ALL_Activation_0611(task_num=10, binary=False, train='train', path='data/0619_OutliersDetection_data/')

    # # robust_dir_path = 'MNIST_robust_result/'
    # # draw_robust_result(robust_dir_path, fv_index=1)
    #
    task_num = 10
    if task_num == 10:
        t = range(1, 15+1)
        dataset = 'MNIST'
        orig_hidden_dim = 50
        samples=50
        SNN_STRUCTURE='15-50-10'

    elif task_num == 20:
        t = range(1, 20 + 1)
        dataset = 'EMNIST20'
        orig_hidden_dim = 67
        samples=50
        SNN_STRUCTURE = '20-67-20'

    elif task_num == 100:
        t=range(1, 50+1)
        dataset = 'CIFAR100'
        samples=10
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'

    # # dirs = 'MNIST_ACC_ChangeDataset_Result/'
    # # draw_changeFV_result(dirs, fv_index=4, task_num=task_num)
    #


    # fv_index_list = [1]#, 2, 3, 4]
    # dirs = 'data/'
    # for fv_index in fv_index_list:
    #     Remake_newFV(dirs, task_num=task_num, fv_index=fv_index, x=1.8)





    # beta = calculate_new_parameter_for_sigmoid(task_num, x=1.8)
    # # print(beta)
    # # # seed=0
    # # # print sigmoid(300, beta)
    # #
    # sample_num = 600
    # sample_list = []
    # t = np.linspace(-5, 5, sample_num)
    # for i in range(sample_num):
    #     sample_list.append(sigmoid(t[i], beta))
    # plt.plot(t, sample_list)
    # plt.hold
    # plt.plot([-1.8, -1.8], [0, 1], 'r', linestyle=':')
    # plt.hold
    # plt.plot([1.8, 1.8], [0, 1], 'r', linestyle=':')
    # plt.subplots_adjust(top=0.94, bottom=0.1, left=0.08, right=0.99, hspace=0, wspace=0)
    # plt.title('%s - Sigmoid(beta=%.5f) Function' % (dataset, beta))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig('%s-NewSigmoidFunction-0619'%(dataset), dpi=300)
    # plt.show()



    #
    #
    # #
    # color_list = ['black', 'gray', 'lightcoral', 'red', 'orangered', 'saddlebrown', 'peru', 'darkorange', 'gold',
    #               'olive',
    #               'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'slategray', 'midnightblue', 'indigo',
    #               'deeppink', 'crimson']
    #


    # fv_index_list = [1]#, 2, 3, 4]
    # beta = calculate_new_parameter_for_sigmoid(task_num, x=1.8)
    # using_sigmoid = 'newsigmoid'
    # for fv_index in fv_index_list:
    #     dir_path = 'data/'
    #     trainset_data, testset_data, diff_trainset_activation_metric, diff_testset_activation_metric = inverse_sigmoid_in_FV(dir_path, fv_index=fv_index, task_num=task_num, beta=beta)
    # #
    #     plt.figure(figsize=(20, 10))
    #     plt.hist(trainset_data, bins=30)
    # #     # plt.hist(trainset_data[500*i:500*(i+1)], bins=30)
    #     plt.title('MNIST FV1 Trainset activations distribution')
    #     plt.ylabel('Numbers')
    #     plt.xlabel('Activation Values')
    #     plt.show()


    #
    #     plt.figure(figsize=(20, 10))
    #     plt.hist(testset_data, bins=30)
    #     plt.ylabel('Numbers')
    #     plt.xlabel('Activation Values')
    # #     plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99, hspace=0, wspace=0)
    # #     # plt.title('{}-FV{}-BeforeSigmoid-({})-Distribution_Result'.format(dataset, fv_index, 'trainset'))
    # #     # plt.savefig('{}-FV{}-BeforeSigmoid-({})-Distribution_Result'.format(dataset, fv_index, 'trainset'))
    # #     plt.title('{}-FV{}-{}-({})-Distribution_Result'.format(dataset, fv_index, using_sigmoid, 'trainset'))
    # #     # plt.savefig('{}-FV{}-{}-({})-Distribution_Result'.format(dataset,  fv_index, using_sigmoid, 'trainset'))

    #     plt.close()
    #
    #     # plt.figure(figsize=(20, 10))
    #     # plt.hist(testset_data[:500], bins=30)
    #     # plt.ylabel('Numbers')
    #     # plt.xlabel('Activation Values')
    #     # plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99, hspace=0, wspace=0)
    #     # plt.title('{}-FV{}-{}-({})-Distribution_Result'.format(dataset, fv_index, using_sigmoid, 'testset'))
    #     # # plt.savefig('{}-FV{}-{}-({})-Distribution_Result'.format(dataset, fv_index, using_sigmoid, 'testset'))
    #     # # # plt.title('{}-FV{}-BeforeSigmoid-({})-Distribution_Result'.format(dataset, fv_index, 'testset'))
    #     # # plt.savefig('{}-FV{}-BeforeSigmoid-({})-Distribution_Result'.format(dataset, fv_index, 'testset'))
    #     # plt.show()
    #     # plt.close()
    #
    #     # plt.figure(figsize=(20, 10))
    #     # for each_task in range(2):
    #     #     for sample in range(samples):
    #     #         plt.plot(t, diff_trainset_activation_metric[each_task, sample, :], color=color_list[each_task])
    #     # plt.show()