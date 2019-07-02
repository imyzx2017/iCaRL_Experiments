import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import json

METHODS = ['ANN Single', 'EWC', 'iCaRL', 'GEM', 'SNN']

def average_accuracy(class_num, file_path):
    result_map_txt = open(file_path, 'r')
    lines = result_map_txt.readlines()
    average_accuracy_array = np.zeros(shape=[class_num])
    learning_step = []
    for id in range(class_num):
        learning_step.append('Step%s' % (id + 1))
    for r, line in enumerate(lines):
        line = line.strip()
        cr_list = line.split(',')[:r + 1]
        cr_array = np.array(cr_list).astype(np.float32)
        average_accuracy_array[r] = cr_array.mean()
        accuracy_var = cr_array.var() / 3
    return average_accuracy_array

def draw_average_accuracy_of_model(task_num, method, root_dir, param_list, is_val=False):
    # model_list = []
    if task_num == 10:
        dataset = 'MNIST'
        interleave = 1
    elif task_num == 20:
        dataset = 'EMNIST20'
        interleave = 1
    elif task_num == 100:
        dataset = 'CIFAR100'
        interleave = 10
    else:
        raise Exception('unsupport task number!')
    save_path = '%s/%s_%s_average_accuracy'%(root_dir, method.replace(' ', ''), dataset)

    acc_array = np.zeros(shape=[task_num, (len(param_list))])
    # var_array = np.zeros(shape=[task_num, (len(model_list))])
    ex_idx = 0
    for p in param_list:
        dst_file = None
        for file in os.listdir(root_dir):
            if '.csv' in file:
                if 'epoch%d'%p in file:
                    dst_file = file
                elif file.split('_')[3] == str(p):
                    dst_file = file
        if dst_file:
            txt = '%s/%s' % (root_dir, dst_file)
            print(txt)
            acc_array[:,ex_idx] = average_accuracy(task_num, txt)
            ex_idx += 1

    acc_array = acc_array[:, :ex_idx]
    # var_array = var_array[:, :ex_idx]

    with open('%s.csv' % save_path, 'w') as f:
        writer = csv.writer(f)
        result_a = acc_array.tolist()
        for line in result_a:
            str_line = [str(x) for x in line]
            writer.writerow(str_line)

    accuracy_array = acc_array.transpose()
    # variance_array = var_array.transpose()

    x = range(1,(task_num+1))
    line_list = []
    plt.figure(figsize=(10, 5))
    for r in range(accuracy_array.shape[0]):
        label = 'epoch ' + str(param_list[r])
        line, = plt.plot(x, accuracy_array[r], label=label)
        # plt.fill_between(x, accuracy_array[r]-variance_array[r], accuracy_array[r]+variance_array[r])
        line_list.append(line)

    plt.legend(handles=line_list, loc=3)

    # plt.xticks(x, fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([1, task_num-1])
    l = np.arange(0, task_num+1 , interleave)
    l = l[1:]

    if task_num==100:
        t2 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.xticks(t2)
    else:
        plt.xticks(l)
    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.title('%s %s average accuracy' % (method, dataset))
    print(save_path)
    plt.savefig(save_path,)
    plt.show()

def draw_average_accuracy_of_model_4FVs(task_num, method, root_dir, is_val=False):
    # model_list = [

    if task_num == 10:
        dataset = 'MNIST'
        interleave = 1
    elif task_num == 20:
        dataset = 'EMNIST20'
        interleave = 1
    elif task_num == 100:
        dataset = 'CIFAR100'
        interleave = 10
    else:
        raise Exception('unsupport task number!')
    net_structure = root_dir[root_dir.find('MLP')+3: -1]
    save_path = '%s/%s_%s_average_accuracy'%(root_dir, method.replace(' ', ''), dataset)

    # acc_array = np.zeros(shape=[4, task_num])
    # var_array = np.zeros(shape=[task_num, (len(model_list))])
    ex_idx = 0


    acc_array_dropout = np.zeros(shape=[4, task_num])
    acc_array_withoutdropout = np.zeros(shape=[4, task_num])
    avg_array_dropout = np.zeros(shape=[4, 1])
    avg_array_withoutdropout = np.zeros(shape=[4, 1])

    dst_file_dropout_list = []
    dst_file_withoutdropout_list = []
    for file in os.listdir(root_dir):
        if '.csv' in file and not '.npz' in file:
            if 'WithoutDropout' in file:
                dst_file_withoutdropout_list.append(file)
            elif 'Dropout' in file:
                dst_file_dropout_list.append(file)
            else:
                pass


    for index, file in enumerate(dst_file_dropout_list):
        txt = '%s/%s' % (root_dir, file)
        print(txt)
        acc_array_dropout[index, :] = average_accuracy(task_num, txt)

    for index, file in enumerate(dst_file_withoutdropout_list):
        txt = '%s/%s' % (root_dir, file)
        print(txt)
        acc_array_withoutdropout[index, :] = average_accuracy(task_num, txt)




    # acc_array = acc_array[:, :, :ex_idx]
    # var_array = var_array[:, :ex_idx]


    ############## Saving Data #################
    for idx, file in enumerate(dst_file_dropout_list):
        current_FV_name = file[file.find('FV'): file.find('FV') + 3]
        np.savetxt('%s_%s_dropout.csv' %(save_path, current_FV_name), np.array(acc_array_dropout[idx, :]))


    for idx, file in enumerate(dst_file_withoutdropout_list):
        current_FV_name = file[file.find('FV'): file.find('FV') + 3]
        np.savetxt('%s_%s_withoutdropout.csv' % (save_path, current_FV_name), np.array(acc_array_withoutdropout[idx, :]))
        # with open('%s_%s_withoutdropout.csv' % (save_path, current_FV_name), 'w') as f:
        #     # acc = acc_array[idx, :, :]
        #     # for item in acc:
        #     #     f.write(item[0])
        #     #     f.write('\n')
        #     # f.close()
        #     writer = csv.writer(f)
        #     result_a = acc_array_dropout[idx, :].tolist()
        #     for line in result_a:
        #         try:
        #             str_line = [str(x) for x in line]
        #         except:
        #             str_line = line
        #         writer.writerow(str(str_line))
    ############################################

    for i in range(avg_array_dropout.shape[0]):
        avg_array_dropout[i] = np.mean(acc_array_dropout[i, :])
    model_avg_save_path = 'result/model_avgacc_avg'
    if not os.path.exists(model_avg_save_path):
        os.mkdir(model_avg_save_path)
    with open('%s/%s.csv' % (model_avg_save_path, net_structure), 'w') as f:
        writer = csv.writer(f)
        result_a = avg_array_dropout.tolist()
        for line in result_a:
            str_line = [str(x) for x in line]
            writer.writerow(str_line)

    ############################
    accuracy_array = acc_array_dropout.transpose()
    # variance_array = var_array.transpose()

    x = range(1,(task_num+1))
    line_list = []


    ########################
    plt.figure(figsize=(10, 5))
    for idx, file in enumerate(dst_file_dropout_list):
        current_FV_name = file[file.find('FV'): file.find('FV') + 3]

        line, = plt.plot(x, acc_array_dropout[idx, :], label=current_FV_name)
        # plt.fill_between(x, accuracy_array[r]-variance_array[r], accuracy_array[r]+variance_array[r])
        line_list.append(line)
        plt.hold

    plt.legend(handles=line_list, loc=3)

    # plt.xticks(x, fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([1, task_num-1])
    l = np.arange(0, task_num+1 , interleave)
    l = l[1:]

    if task_num==100:
        t2 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.xticks(t2)
    else:
        plt.xticks(l)
    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.title('%s %s average accuracy--(%s)--net_struc: %s' % (method, dataset, 'Using Dropout', net_structure))
    print(save_path)
    plt.savefig(save_path + '_dropout', )
    plt.show()
    plt.close()


    #####################
    plt.figure(figsize=(10, 5))
    for idx, file in enumerate(dst_file_withoutdropout_list):
        current_FV_name = file[file.find('FV'): file.find('FV') + 3]

        line, = plt.plot(x, acc_array_withoutdropout[idx, :], label=current_FV_name)
        # plt.fill_between(x, accuracy_array[r]-variance_array[r], accuracy_array[r]+variance_array[r])
        line_list.append(line)
        plt.hold

    # plt.legend(handles=line_list, loc=3)

    # plt.xticks(x, fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([1, task_num-1])
    plt.legend()
    l = np.arange(0, task_num+1 , interleave)
    l = l[1:]
    if task_num==100:
        t2 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.xticks(t2)
    else:
        plt.xticks(l)
    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.title('%s %s average accuracy--(%s)--net_struc: %s' % (method, dataset, 'Without Dropout', net_structure))
    print(save_path)
    plt.savefig(save_path+'_withoutdropout')
    plt.show()
    ###############################

def average_accuracy_from_txt(class_num, file_path):
    result_map_txt = open(file_path, 'r')
    lines = result_map_txt.readlines()
    average_accuracy_array = np.zeros(shape=[class_num])
    learning_step = []
    for id in range(class_num):
        learning_step.append('Step%s' % (id + 1))
    for r, line in enumerate(lines):
        average_accuracy_array[r] = float(line.split(' ')[-1])
    return average_accuracy_array

def draw_average_accuracy_of_snn(task_num, method, root_dir):
    model_list = []
    if task_num == 10:
        dataset = 'MNIST'
        interleave = 1
    elif task_num == 20:
        dataset = 'EMNIST20'
        interleave = 1
    elif task_num == 100:
        dataset = 'CIFAR100'
        interleave = 10
    else:
        raise Exception('unsupport task number!')
    dir_list = os.listdir(root_dir)
    dataset_num = 4
    acc_array = np.zeros(shape=[task_num, dataset_num])
    ex_idx = 0
    fv_name_list = []
    for cur_dir in dir_list:
        if '.' in cur_dir:
            continue
        dst_file = None
        for file in os.listdir(os.path.join(root_dir, cur_dir, 'ANN_final_result')):
            if ('FINAL_SNN_CL_result') in file:
                dst_file = file
        if dst_file:
            txt = os.path.join(root_dir, cur_dir, 'ANN_final_result', dst_file)
            print(txt)
            acc_array[:,ex_idx] = average_accuracy_from_txt(task_num, txt)
            fv_name = cur_dir
            fv_name_list.append(fv_name)
            ex_idx += 1

    save_path = '%s/%s_%s_average_accuracy'%(root_dir, method.replace(' ', ''), dataset)

    acc_array = acc_array[:,:ex_idx]

    with open('%s.csv'%save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        result_a = acc_array.tolist()
        for line in result_a:
            str_line = [str(x) for x in line]
            writer.writerow(str_line)

    accuracy_array = acc_array.transpose()

    x = range(1,(task_num+1))
    line_list = []
    plt.figure(figsize=(10, 5))
    for r in range(accuracy_array.shape[0]):
        line, = plt.plot(x, accuracy_array[r], label=fv_name_list[r])
        line_list.append(line)

    plt.legend(handles=line_list, loc=3)

    plt.ylim([0, 1])
    plt.xlim([1, task_num-1])
    l = np.arange(0, task_num+1 , interleave)
    l = l[1:]
    plt.xticks(ticks=l)
    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.title('%s %s average accuracy' % (method, dataset))
    plt.savefig(save_path)
    plt.show()

def plotting_diff_model_avgacc(dir, task_num, fv_index, hidden_num=100):
    plt.figure(figsize=(12, 8))
    # color_list = ['g', 'c', 'm', 'y', 'k', 'darkviolet', 'midnightblue', 'peru', 'deepskyblue', 'darkorchid', 'brown', 'deeppink', 'black', 'coral',
    #               'chartreuse', 'yellow', 'darkorange', 'indigo']

    color_list = ['black', 'gray', 'lightcoral', 'red', 'orangered', 'saddlebrown', 'peru', 'darkorange', 'gold', 'olive',
                  'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'slategray', 'midnightblue', 'indigo', 'deeppink', 'crimson']
    t = range(1, (task_num + 1))
    plot_num = 0

    if task_num == 10:
        dataset = 'MNIST'
        orig_hidden_dim = 50
        SNN_STRUCTURE='15-50-10'

    elif task_num == 20:
        dataset = 'EMNIST20'
        orig_hidden_dim = 67
        SNN_STRUCTURE = '20-67-20'

    elif task_num == 100:
        dataset = 'CIFAR100'
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'

    SNN_result_dir = 'SNN_result/'
    SNN_RESULT = Load_SNN_Result(SNN_result_dir, task_num, fv_index=fv_index, neuron_model='GC')

    SNN_CA3_RESULT = Load_SNN_Result(SNN_result_dir, task_num, fv_index=fv_index, neuron_model='CA3')


    ###################### plotting MLP width change width result ##########################
    for each_model_dir in os.listdir(dir):
        if 'MLP' in each_model_dir:
            current_model_dir = dir + each_model_dir + '/'
            current_model_structure = each_model_dir[each_model_dir.find('MLP'):]
            if len(current_model_structure.split('-'))==3 and int(current_model_structure.split('-')[-1])==task_num:
                for file in os.listdir(current_model_dir):
                    if '.csv' in file and 'FV%s'%fv_index in file and 'withoutdropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = current_model_structure + ',withoutdropout'
                        # if plot_num>=10:
                        #     if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label, marker='o')
                        #     else:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label)
                        # else:
                        if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label, marker='o')
                        else:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label)
                        plt.hold
                        plot_num+=1
                    elif '.csv' in file and 'FV%s'%fv_index in file and 'dropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = current_model_structure + ',dropout(0.5)'
                        # if plot_num>=10:
                        #     if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label)#, linestyle='-.')
                        #     else:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label)
                        # else:
                        if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label)#, linestyle='-.')
                        else:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label)
                        plt.hold
                        plot_num += 1
                    else:
                        pass

    plt.plot(t, SNN_RESULT, color=color_list[int(plot_num / 2)], marker='*', label='SNN-GC-(%s)WithoutUsingMemory' % SNN_STRUCTURE)
    plot_num+=2
    plt.hold


    plt.plot(t, SNN_CA3_RESULT, color=color_list[int(plot_num / 2)], marker='*',
             label='SNN-CA3-(%s)WithoutUsingMemory' % SNN_STRUCTURE)
    plot_num+=1

    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.ylim([0, 1.1])
    if task_num==100:
        t2 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.xticks(t2)
    else:

        plt.xticks(t)
    plt.title('%s--Change MLP hidden layer (width) in iCaRL, FV%s, average accuracy'%(dataset, fv_index))
    plt.legend()
    plt.savefig('result/%s_change_model_width_FV%d_result.png'%(dataset, fv_index))
    plt.show()
    #############################################################################

    ###################### plotting MLP width change Depth result ##########################
    plt.figure(figsize=(12, 8))
    color_num_list = []
    temp_dropout_num = 0
    temp_nodropout_num = 0
    for idx, each_model_dir in enumerate(os.listdir(dir)):
        if 'MLP' in each_model_dir:
            current_model_dir = dir + each_model_dir + '/'
            current_model_structure = each_model_dir[each_model_dir.find('MLP'):]
            if not len(current_model_structure.split('-'))==3 and int(current_model_structure.split('-')[-1])==task_num:
                input_dim, output_dim = current_model_structure.split('-')[0], current_model_structure.split('-')[-1]
                hidden_dim = current_model_structure.split('-')[1]
                current_mlp_hiddenlayer_number = len(current_model_structure.split('-')) - 2
                for file in os.listdir(current_model_dir):
                    if '.csv' in file and 'FV%s'%fv_index in file and 'withoutdropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',withoutdropout'

                        plt.plot(t, current_model_result, color=color_list[int(color_num_list[temp_nodropout_num])*4], label=current_label)

                        plt.hold
                        temp_nodropout_num+=1
                    elif '.csv' in file and 'FV%s'%fv_index in file and 'dropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',dropout(0.5)'

                        color_num_list.append(temp_dropout_num)
                        plt.plot(t, current_model_result, color=color_list[int(color_num_list[temp_nodropout_num])*4], label=current_label, linestyle=':')

                        plt.hold
                        temp_dropout_num += 1
                    else:
                        pass

            #########  plotting 1 layers result  ##################
            elif int(current_model_structure.split('-')[1])==hidden_num and int(current_model_structure.split('-')[-1])==task_num:
                input_dim, output_dim = current_model_structure.split('-')[0], current_model_structure.split('-')[-1]
                hidden_dim = current_model_structure.split('-')[1]
                current_mlp_hiddenlayer_number = len(current_model_structure.split('-')) - 2
                for file in os.listdir(current_model_dir):
                    if '.csv' in file and 'FV%s'%fv_index in file and 'withoutdropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',withoutdropout'

                        plt.plot(t, current_model_result, color=color_list[-7], label=current_label)

                        plt.hold
                        plot_num+=1
                    elif '.csv' in file and 'FV%s'%fv_index in file and 'dropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',dropout(0.5)'

                        plt.plot(t, current_model_result, color=color_list[-7], label=current_label, linestyle=':')

                        plt.hold
                        plot_num += 1
                    else:
                        pass

    plt.plot(t, SNN_RESULT, color=color_list[-2], marker='*', label='SNN-GC-(%s)WithoutUsingMemory'%SNN_STRUCTURE)
    plt.hold
    plot_num += 2

    plt.plot(t, SNN_CA3_RESULT, color=color_list[-1], marker='*',
                 label='SNN-CA3-(%s)WithoutUsingMemory' % SNN_STRUCTURE)
    plot_num += 1

    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.ylim([0,1.1])

    if task_num==100:
        t2 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.xticks(t2)
    else:
        plt.xticks(t)
    plt.title('%s--Change MLP hidden layer (Depth) in iCaRL, FV%s, average accuracy'%(dataset, fv_index))
    plt.legend()
    plt.savefig('result/%s_change_model_depth_FV%d_result.png'%(dataset, fv_index))
    plt.show()

def Load_SNN_Result(dir, task_num, fv_index, neuron_model='GC'):
    result = np.zeros(shape=[task_num])
    saving_path = 'result/SNN_Result/'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    if task_num == 10:
        dataset = 'MNIST10'
        orig_hidden_dim = 50

    elif task_num == 20:
        dataset = 'EMNIST20'
        orig_hidden_dim = 67
    elif task_num == 100:
        dataset = 'CIFAR100'
        orig_hidden_dim = 100

    for each_dataset_dir in os.listdir(dir):
        if each_dataset_dir.split('-')[0]==dataset and not '.rar' in each_dataset_dir:
            for neuron_model_dir in os.listdir(dir + each_dataset_dir):
                if neuron_model_dir==neuron_model:

                    ####
                    each_dataset_dir = each_dataset_dir + '/' + neuron_model_dir
                    for each_fv_dir in os.listdir(dir + each_dataset_dir):
                        if str(fv_index) in each_fv_dir:
                            for sub_dir in os.listdir(dir + each_dataset_dir + '/' + each_fv_dir):
                                if sub_dir=='SNN_final_result':
                                    for file in os.listdir(dir + each_dataset_dir + '/' + each_fv_dir + '/' + sub_dir):
                                        if 'FINAL_SNN_CL' in file:
                                            path = dir + each_dataset_dir + '/' + each_fv_dir + '/' + sub_dir + '/'
                                            f = open(path + file)
                                            temp = f.readlines()
                                            f.close()
                                            for i, item in enumerate(temp):
                                                result[i] = float(item.split(' ')[-1])
    np.savetxt(saving_path + '%s_FV%s_result.csv' % (dataset, fv_index), result)

    return result

def Load_SNN_UsingMemoryResult(dir, task_num, fv_index, neuron_model='GC', Memory=1):
    result = np.zeros(shape=[task_num])
    saving_path = 'result/SNN_Result/'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    if task_num == 10:
        dataset = 'MNIST10'
        orig_hidden_dim = 50

    elif task_num == 20:
        dataset = 'EMNIST20'
        orig_hidden_dim = 67
    elif task_num == 100:
        dataset = 'CIFAR100'
        orig_hidden_dim = 100

    for each_dataset_dir in os.listdir(dir):
        if each_dataset_dir.split('-')[0]==dataset and not '.rar' in each_dataset_dir and 'MEMORY({})'.format(Memory) in each_dataset_dir:
            for neuron_model_dir in os.listdir(dir + each_dataset_dir):
                if neuron_model_dir==neuron_model:

                    ####
                    each_dataset_dir = each_dataset_dir + '/' + neuron_model_dir
                    for each_fv_dir in os.listdir(dir + each_dataset_dir):
                        if str(fv_index) in each_fv_dir:
                            for sub_dir in os.listdir(dir + each_dataset_dir + '/' + each_fv_dir):
                                for file in os.listdir(dir + each_dataset_dir + '/' + each_fv_dir + '/' + sub_dir):
                                    if 'FINAL_SNN_CL' in file:
                                        path = dir + each_dataset_dir + '/' + each_fv_dir + '/' + sub_dir + '/'
                                        f = open(path + file)
                                        temp = f.readlines()
                                        f.close()
                                        for i, item in enumerate(temp):
                                            result[i] = float(item.split(' ')[-1])
    np.savetxt(saving_path + '%s_FV%s_Using(%s)Memory_result.csv' % (dataset, fv_index, Memory), result)

    return result

def Load_SNN_fewshot_Result(dir, task_num, fv_index, neuron_model='GC', traintest = 'TRAIN10TEST50'):
    result = np.zeros(shape=[task_num])
    saving_path = 'result/SNN_fewshot_Result/'
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    if task_num == 10:
        dataset = 'MNIST10'
        orig_hidden_dim = 50

    elif task_num == 20:
        dataset = 'EMNIST20'
        orig_hidden_dim = 67
    elif task_num == 100:
        dataset = 'CIFAR100'
        orig_hidden_dim = 100

    for each_dataset_dir in os.listdir(dir):
        if each_dataset_dir.split('-')[0]==dataset and not '.rar' in each_dataset_dir:
            for neuron_model_dir in os.listdir(dir + each_dataset_dir):
                if neuron_model_dir==neuron_model:

                    ####
                    each_dataset_dir = each_dataset_dir + '/' + neuron_model_dir
                    for train_test in os.listdir(dir + each_dataset_dir):
                        if train_test==traintest:
                            each_dataset_dir = each_dataset_dir + '/' + train_test
                            for each_fv_dir in os.listdir(dir + each_dataset_dir):
                                if str(fv_index) in each_fv_dir:
                                    for sub_dir in os.listdir(dir + each_dataset_dir + '/' + each_fv_dir):
                                        for file in os.listdir(dir + each_dataset_dir + '/' + each_fv_dir + '/' + sub_dir):
                                            if 'FINAL_SNN_CL' in file:
                                                path = dir + each_dataset_dir + '/' + each_fv_dir + '/' + sub_dir + '/'
                                                f = open(path + file)
                                                temp = f.readlines()
                                                f.close()
                                                for i, item in enumerate(temp):
                                                    result[i] = float(item.split(' ')[-1])
    np.savetxt(saving_path + '%s_FV%s_result.csv' % (dataset, fv_index), result)

    return result

def draw_bias_difference(dir, fv_index, task_num=20):
    plt.figure(figsize=(10, 5))
    t = range(1, (task_num + 1))
    for bias_dir in os.listdir(dir):
        if 'nobias' in bias_dir:
            path = dir + bias_dir + '\\'
            bias_name = 'nobias'
        elif 'withbias' in bias_dir:
            path = dir + bias_dir + '\\'
            bias_name = 'withbias'
        for file in os.listdir(path):
            if '.csv' in file and 'FV{}'.format(fv_index) in file and 'average' in file:
                if 'withoutdropout' in file:
                    drop_name = 'withoutdropout'
                else:
                    drop_name = 'dropout'
                file_path = path + file
                data = np.loadtxt(file_path)
                plt.plot(t, data, label='EMNIST20_20-67-20_FV{}_{}_{}'.format(fv_index, bias_name, drop_name))
                plt.hold

    plt.xticks(t)
    plt.ylim([0, 1.1])
    plt.legend()
    plt.show()

def draw_patience_difference(dir, fv_index, task_num=20):
    plt.figure(figsize=(10, 5))
    t = range(1, (task_num + 1))
    for bias_dir in os.listdir(dir):
        if 'nobias' in bias_dir:
            path = dir + bias_dir + '\\'
            bias_name = 'nobias'
            pass
        elif 'withbias' in bias_dir:
            path = dir + bias_dir + '\\'
            bias_name = 'withbias'
            patience = int(bias_dir.split('_')[-1][1:])
            for file in os.listdir(path):
                if '.csv' in file and 'FV{}'.format(fv_index) in file and 'average' in file:
                    if 'withoutdropout' in file:
                        drop_name = 'withoutdropout'
                        file_path = path + file
                        data = np.loadtxt(file_path)
                        plt.plot(t, data, label='EMNIST20_20-67-20_FV{}_{}_{},patience:{}'.format(fv_index, bias_name, drop_name, patience))
                        plt.hold

                    else:
                        drop_name = 'dropout'

    plt.xticks(t)
    plt.ylim([0, 1.1])
    plt.legend()
    plt.show()

def draw_diff_samples_acc_result(dir, fv_index, dataset_nums=3, task_num=20, drop='withoutdropout'):

    if task_num == 10:
        dataset = 'MNIST'
        orig_hidden_dim = 50

    elif task_num == 20:
        dataset = 'EMNIST20'
        orig_hidden_dim = 67

    elif task_num == 100:
        dataset = 'CIFAR100'
        orig_hidden_dim = 100
    t = range(1, (task_num + 1))

    if task_num==100:
        color_list = ['blue', 'r', 'g']
    else:
        if dataset_nums==4:
            color_list = ['pink', 'r', 'g', 'blue', 'yellow', 'orange', 'cyan']
        else:
            color_list = ['r', 'g', 'blue', 'pink', 'yellow', 'orange', 'cyan']
    nodrop_data_index = 0
    drop_data_index = 0

    plt.figure(figsize=(10, 5))
    for dir1 in os.listdir(dir):
        if 'train' in dir1:
            name = dir1.split('-')[-1]
            temp_list = name.split('train')[-1].split('test')
            if not temp_list[0]==temp_list[1]:
                use_name = '(fewshotlearning)'
            else:
                use_name = ''
            path = dir + dir1 + '\\'
            for dataset_dir in os.listdir(path):
                if dataset_dir.split('_')[0] == dataset:
                    net_struc = dataset_dir.split('MLP')[-1]
                    path = path + dataset_dir + '\\'
                    for each_file in os.listdir(path):
                        if '.csv' in each_file and 'average' in each_file and 'FV{}'.format(fv_index) in each_file and 'withoutdropout' in each_file:
                            if drop in each_file:
                                data = np.loadtxt(path + each_file)
                                plt.plot(t, data, label='{}_MLP_{}_FV{}_{}_{}{}'.format(dataset, net_struc, fv_index, 'nodropout', name, use_name), color=color_list[nodrop_data_index])
                                plt.hold
                                nodrop_data_index+=1

                        elif '.csv' in each_file and 'average' in each_file and 'FV{}'.format(fv_index) in each_file and 'dropout' in each_file:
                            if drop in each_file:
                                data = np.loadtxt(path + each_file)
                                plt.plot(t, data,
                                         label='{}_MLP_{}_FV{}_{}_{}{}'.format(dataset, net_struc, fv_index, 'dropout',
                                                                               name, use_name), color=color_list[drop_data_index], linestyle=':')
                                plt.hold
                                drop_data_index += 1

    # add SNN result
    SNN_fewshot_result_dir = 'SNN_fewshot_result/'

    if not task_num==100:
        SNN_train10test50_RESULT = Load_SNN_fewshot_Result(SNN_fewshot_result_dir, task_num, fv_index=fv_index, neuron_model='GC', traintest='TRAIN10TEST50')
        SNN_train30test50_RESULT = Load_SNN_fewshot_Result(SNN_fewshot_result_dir, task_num, fv_index=fv_index, neuron_model='GC',
                                                           traintest='TRAIN30TEST50')
        plt.plot(t, SNN_train10test50_RESULT,
                 label='{}_SNN({})_{}_FV{}_{}(fewshot)'.format(dataset, 'GC', net_struc, fv_index,
                                                       'train10test50'), color='orangered', marker='*')
        plt.hold

        plt.plot(t, SNN_train30test50_RESULT,
                 label='{}_SNN({})_{}_FV{}_{}(fewshot)'.format(dataset, 'GC', net_struc, fv_index,
                                                              'train30test50'), color='yellowgreen', marker='*')
        plt.hold

        SNN_train50test50_result_dir = 'SNN_result/'
        SNN_train50test50_RESULT = Load_SNN_Result(SNN_train50test50_result_dir, task_num, fv_index=fv_index, neuron_model='GC')
        plt.plot(t, SNN_train50test50_RESULT, label='{}_SNN({})_{}_FV{}_{}'.format(dataset, 'GC', net_struc, fv_index,
                                                              'train50test50'), color='darkblue', marker='*')

    else:
        SNN_train3test10_RESULT = Load_SNN_fewshot_Result(SNN_fewshot_result_dir, task_num, fv_index=fv_index,
                                                           neuron_model='GC', traintest='TRAIN3TEST10')
        SNN_train6test10_RESULT = Load_SNN_fewshot_Result(SNN_fewshot_result_dir, task_num, fv_index=fv_index,
                                                           neuron_model='GC',
                                                           traintest='TRAIN6TEST10')
        plt.plot(t, SNN_train3test10_RESULT,
                 label='{}_SNN({})_{}_FV{}_{}(fewshot)'.format(dataset, 'GC', net_struc, fv_index,
                                                               'train3test10'), color='orangered', marker='*')
        plt.hold
        plt.plot(t, SNN_train6test10_RESULT,
                 label='{}_SNN({})_{}_FV{}_{}(fewshot)'.format(dataset, 'GC', net_struc, fv_index,
                                                               'train6test10'), color='yellowgreen', marker='*')
        plt.hold
        SNN_train10test10_result_dir = 'SNN_result/'
        SNN_train10test10_RESULT = Load_SNN_Result(SNN_train10test10_result_dir, task_num, fv_index=fv_index,
                                                   neuron_model='GC')
        plt.plot(t, SNN_train10test10_RESULT, label='{}_SNN({})_{}_FV{}_{}'.format(dataset, 'GC', net_struc, fv_index,
                                                                                   'train10test10'), color='darkblue',
                 marker='*')




    plt.ylim([0, 1.1])

    if task_num==100:
        plt.xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    else:
        plt.xticks(t)
    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.title('%s %s average accuracy, FV%s, net_struc: %s' % ('iCaRL', dataset, fv_index, net_struc))
    plt.legend()
    plt.savefig(dir + 'result\\{}_MLP_{}_change_traintest_result_FV{}_{}.png'.format(dataset, net_struc, fv_index, drop), dpi=300)
    # plt.show()

def plotting_diff_model_avgacc_WithSNN_MemoryResult(dir, task_num, fv_index, hidden_num=100):
    plt.figure(figsize=(20, 10))
    # color_list = ['g', 'c', 'm', 'y', 'k', 'darkviolet', 'midnightblue', 'peru', 'deepskyblue', 'darkorchid', 'brown', 'deeppink', 'black', 'coral',
    #               'chartreuse', 'yellow', 'darkorange', 'indigo']

    color_list = ['black', 'gray', 'lightcoral', 'red', 'orangered', 'saddlebrown', 'peru', 'darkorange', 'gold', 'olive',
                  'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'crimson', 'midnightblue', 'indigo', 'deeppink', 'crimson', 'darkviolet','coral']
    t = range(1, (task_num + 1))
    plot_num = 0

    if task_num == 10:
        dataset = 'MNIST'
        orig_hidden_dim = 50
        SNN_STRUCTURE='15-50-10'

    elif task_num == 20:
        dataset = 'EMNIST20'
        orig_hidden_dim = 67
        SNN_STRUCTURE = '20-67-20'

    elif task_num == 100:
        dataset = 'CIFAR100'
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'

    SNN_result_dir = 'SNN_UsingOneMemory_Result/'
    if not task_num==100:
        SNN_Memory_One_RESULT = Load_SNN_UsingMemoryResult(SNN_result_dir, task_num, fv_index=fv_index, neuron_model='GC', Memory=1)

    SNN_No_Memory_RESULT = Load_SNN_UsingMemoryResult(SNN_result_dir, task_num, fv_index=fv_index, neuron_model='GC',
                                                       Memory=0)

    SNN_NoLearning_RESULT_dir = 'SNN_NoLearning_Result/'
    SNN_NoLearning_RESULT = Load_SNN_Result(SNN_NoLearning_RESULT_dir, task_num, fv_index=fv_index, neuron_model='GC')


    # SNN_CA3_RESULT = Load_SNN_Result(SNN_result_dir, task_num, fv_index=fv_index, neuron_model='CA3')


    ###################### plotting MLP width change width result ##########################
    for each_model_dir in os.listdir(dir):
        if 'MLP' in each_model_dir:
            current_model_dir = dir + each_model_dir + '/'
            current_model_structure = each_model_dir[each_model_dir.find('MLP'):]
            if len(current_model_structure.split('-'))==3 and int(current_model_structure.split('-')[-1])==task_num:
                for file in os.listdir(current_model_dir):
                    if '.csv' in file and 'FV%s'%fv_index in file and 'withoutdropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = current_model_structure + ',withoutdropout'
                        # if plot_num>=10:
                        #     if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label, marker='o')
                        #     else:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label)
                        # else:
                        if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label)#, marker='o')
                        else:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label)
                        plt.hold
                        plot_num+=1
                    elif '.csv' in file and 'FV%s'%fv_index in file and 'dropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = current_model_structure + ',dropout(0.5)'
                        # if plot_num>=10:
                        #     if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label)#, linestyle='-.')
                        #     else:
                        #         plt.plot(t, current_model_result, color=color_list[plot_num-10], label=current_label)
                        # else:
                        if int(current_model_structure.split('-')[1])==orig_hidden_dim:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label)#, linestyle='-.')
                        else:
                            plt.plot(t, current_model_result, color=color_list[plot_num], label=current_label)
                        plt.hold
                        plot_num += 1
                    else:
                        pass



    if task_num==100:
        plot_num += 2
    # plt.plot(t, SNN_No_Memory_RESULT, color=color_list[int(plot_num / 2)], marker='*',
    #          label='SNN-GC-(%s)WithoutUsingMemory' % SNN_STRUCTURE)
    # plot_num+=2
    # plt.hold
    #
    # plt.plot(t, SNN_NoLearning_RESULT, color=color_list[int(plot_num / 2)], marker='*',
    #          label='SNN-GC-(%s)-WithoutLearning' % SNN_STRUCTURE)
    # plot_num += 2
    # plt.hold
    #
    # if not task_num == 100:
    #     plt.plot(t, SNN_Memory_One_RESULT, color=color_list[int(plot_num / 2)], marker='*',
    #              label='SNN-GC-(%s)Using(One)Memory' % SNN_STRUCTURE)
    #     plot_num += 2
    #     plt.hold

    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99, hspace=0, wspace=0)
    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.ylim([0, 1.1])

    if task_num==100:
        plt.xlim([0, task_num + 1])
        t2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.xticks(t2)
    else:

        plt.xticks(t)
    plt.title('%s--Change MLP hidden layer (width) in iCaRL, FV%s, average accuracy'%(dataset, fv_index))
    plt.legend(loc=3)
    plt.savefig('result/%s_change_model_width_FV%d_result.png'%(dataset, fv_index), dpi=300)
    plt.show()
    #############################################################################

    ###################### plotting MLP width change Depth result ##########################
    plt.figure(figsize=(20, 10))
    color_num_list = []
    temp_dropout_num = 0
    temp_nodropout_num = 0
    for idx, each_model_dir in enumerate(os.listdir(dir)):
        if 'MLP' in each_model_dir:
            current_model_dir = dir + each_model_dir + '/'
            current_model_structure = each_model_dir[each_model_dir.find('MLP'):]
            if not len(current_model_structure.split('-'))==3 and int(current_model_structure.split('-')[-1])==task_num:
                input_dim, output_dim = current_model_structure.split('-')[0], current_model_structure.split('-')[-1]
                hidden_dim = current_model_structure.split('-')[1]
                current_mlp_hiddenlayer_number = len(current_model_structure.split('-')) - 2
                for file in os.listdir(current_model_dir):
                    if '.csv' in file and 'FV%s'%fv_index in file and 'withoutdropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',withoutdropout'

                        plt.plot(t, current_model_result, color=color_list[int(color_num_list[temp_nodropout_num])*4], label=current_label)

                        plt.hold
                        temp_nodropout_num+=1
                    elif '.csv' in file and 'FV%s'%fv_index in file and 'dropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',dropout(0.5)'

                        color_num_list.append(temp_dropout_num)
                        plt.plot(t, current_model_result, color=color_list[int(color_num_list[temp_nodropout_num])*4], label=current_label, linestyle=':')

                        plt.hold
                        temp_dropout_num += 1
                    else:
                        pass

            #########  plotting 1 layers result  ##################
            elif int(current_model_structure.split('-')[1])==hidden_num and int(current_model_structure.split('-')[-1])==task_num:
                input_dim, output_dim = current_model_structure.split('-')[0], current_model_structure.split('-')[-1]
                hidden_dim = current_model_structure.split('-')[1]
                current_mlp_hiddenlayer_number = len(current_model_structure.split('-')) - 2
                for file in os.listdir(current_model_dir):
                    if '.csv' in file and 'FV%s'%fv_index in file and 'withoutdropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',withoutdropout'

                        plt.plot(t, current_model_result, color=color_list[-7], label=current_label)

                        plt.hold
                        plot_num+=1
                    elif '.csv' in file and 'FV%s'%fv_index in file and 'dropout' in file:
                        current_model_result = np.loadtxt(current_model_dir + file)
                        current_label = '%s-%s[%sHiddenLayers]-%s'%(input_dim, hidden_dim, current_mlp_hiddenlayer_number, output_dim) + ',dropout(0.5)'

                        plt.plot(t, current_model_result, color=color_list[-7], label=current_label, linestyle=':')

                        plt.hold
                        plot_num += 1
                    else:
                        pass


    # if task_num==100:
    #     plot_num += 2
    # plt.plot(t, SNN_No_Memory_RESULT, color=color_list[-1], marker='*',
    #              label='SNN-GC-(%s)WithoutUsingMemory' % SNN_STRUCTURE)
    # plot_num += 2
    # plt.hold
    # plt.plot(t, SNN_NoLearning_RESULT, color=color_list[int(plot_num / 2)], marker='*',
    #          label='SNN-GC-(%s)-WithoutLearning' % SNN_STRUCTURE)
    # plot_num += 2
    # plt.hold
    #
    # if not task_num == 100:
    #     plt.plot(t, SNN_Memory_One_RESULT, color=color_list[int(plot_num / 2)], marker='*',
    #              label='SNN-GC-(%s)Using(One)Memory' % SNN_STRUCTURE)
    #     plot_num += 2
    #     plt.hold

    plt.ylabel('Average accuracy')
    plt.xlabel('Number of tasks')
    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.99, hspace=0, wspace=0)
    plt.ylim([0, 1.1])

    if task_num==100:
        plt.xlim([0, task_num + 1])
        t2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        plt.xticks(t2)
    else:
        plt.xticks(t)

    plt.title('%s--Change MLP hidden layer (Depth) in iCaRL, FV%s, average accuracy'%(dataset, fv_index))
    plt.legend(loc=3)
    plt.savefig('result/%s_change_model_depth_FV%d_result.png'%(dataset, fv_index), dpi=300)
    plt.show()

if __name__ == '__main__':
    task_num = 10
    method = METHODS[2]

    if task_num == 20:
        input_dim = 20
        output_dim = 20
        dataset = 'EMNIST20'
    elif task_num == 10:
        input_dim = 15
        output_dim = 10
        dataset = 'MNIST'
    elif task_num == 100:
        input_dim = 50
        output_dim = 100
        dataset = 'CIFAR100'

    if method == 'SNN':
        root_dir = 'D:\\vacation\continue-learn\ANNSingle-master\data\\raw\CIFAR100-0408'
        draw_average_accuracy_of_snn(task_num, method, root_dir)
    else:
        # param_list = [100,200,300,400]
        # param_list = [5000]
        # root_dir = 'D:\\vacation\continue-learn\GEM-master\\results/'


        # hidden_width = [50, 67, 100, 200, 400, 800, 1600]
        # hidden_width = [[100, 100], [100, 100, 100], [100, 100, 100, 100]]
        #
        # # hidden_width = [20, 50, 100, 200, 400, 800, 1600]
        # # hidden_width = [3200]

        # hidden_width = [[20], [50], [100], [200], [400], [800], [1600]]     # for MNIST
        # hidden_width = [[2400], [3200]]  # for MNIST

        # hidden_width = [[400, 400], [400, 400, 400]]
        # # #
        # # #
        # hidden_width = [[20], [50], [67], [100], [200], [400], [800], [1600], [2400], [3200]]  # for EMNIST
        # # #
        # # # # hidden_width = [[20], [50], [100], [200], [400], [800], [1600], [2400], [3200]]  # for EMNIST

        # hidden_width = [[50], [167], [200], [400], [800], [1600]]     # for CIFAR100

        # hidden_width = [[15]]
        #

        # hidden_width = [[50], [50, 50], [50, 50, 50]]
        # hidden_width = [[67], [67, 67], [67, 67, 67]]
        #
        # for hidden_num in hidden_width:
        #     if isinstance(hidden_num, int):
        #         root_dir = 'result/%s_FV_MLP%s-%s-%s/'%(dataset, input_dim, hidden_num, output_dim)
        #     else:
        #         hidden_str = ''
        #         for item in hidden_num:
        #             hidden_str += str(item) + '-'
        #         root_dir = 'result/%s_FV_MLP%s-%s%s/'%(dataset, input_dim, hidden_str, output_dim)
        #
        #     draw_average_accuracy_of_model_4FVs(task_num, method, root_dir, is_val=False)


        hidden_width = [[15]]#, [20], [50], [67], [100], [200], [400], [800], [1600], [2400], [3200]]  # for EMNIST


        # hidden_width = [[15]]
        # # dir_list = ['MNIST10-NOISE(0.3)','MNIST10-NOISE(0.5)','MNIST10-NOISE(0.7)']
        # dir_list = ['FVnew-train50test50-cleaned']
        dir_list = ['CMNIST-NEW-BINARY-FV1']
        for dir_temp in dir_list:
            for hidden_num in hidden_width:
                if isinstance(hidden_num, int):
                    root_dir = 'result/%s/%s_FV_MLP%s-%s-%s/' % (dir_temp, dataset, input_dim, hidden_num, output_dim)
                else:
                    hidden_str = ''
                    for item in hidden_num:
                        hidden_str += str(item) + '-'
                    root_dir = 'result/%s/%s_FV_MLP%s-%s%s/' % (dir_temp, dataset, input_dim, hidden_str, output_dim)

                draw_average_accuracy_of_model_4FVs(task_num, method, root_dir, is_val=False)



        # train_test = ['FV-train3test10', 'FV-train6test10']

        # train_test = ['FV-train2test50'] #, 'FV-train10test50']
        # for each_dir in train_test:
        #     for hidden_num in hidden_width:
        #         if isinstance(hidden_num, int):
        #             root_dir = 'result/%s/%s_FV_MLP%s-%s-%s/' % (each_dir, dataset, input_dim, hidden_num, output_dim)
        #         else:
        #             hidden_str = ''
        #             for item in hidden_num:
        #                 hidden_str += str(item) + '-'
        #             root_dir = 'result/%s/%s_FV_MLP%s-%s%s/' % (each_dir, dataset, input_dim, hidden_str, output_dim)
        #
        #         draw_average_accuracy_of_model_4FVs(task_num, method, root_dir, is_val=False)

        # hidden_width = [[20], [50], [67], [100], [200], [400], [800], [1600], [2400], [3200]]  # for EMNIST



        # hidden_width = [[20], [50], [67], [100], [200], [400], [800], [1600], [2400], [3200]]  # for EMNIST\

        # # dir_list = ['FV-train50test50']
        # dir_list = ['FVnew-train50test50-0619']
        # # # dir_list = ['MNIST10-NOISE(0.3)']#,'MNIST10-NOISE(0.5)','MNIST10-NOISE(0.7)']
        # result_dir = 'result/'
        # fv_list = [1]#, 2, 3, 4]
        #
        # for temp_dir in dir_list:
        #     for fv_id in fv_list:
        #         result_dir = result_dir + temp_dir + '/'
        #     # # hidden_num = 800 for MNIST; 400 for EMNIST;1600 for CIFAR100
        #         plotting_diff_model_avgacc_WithSNN_MemoryResult(result_dir, task_num, fv_index=fv_id, hidden_num=67)
        #     #     result_dir = 'result/'
        #
        #         # plotting_diff_model_avgacc(result_dir, task_num, fv_index=fv_id, hidden_num=400)

        # SNN_result_dir = 'SNN_result/'
        # Load_SNN_Result(SNN_result_dir, task_num, fv_index=1)


        # compare_dir = 'D:\\Projects\\Projects\\pytorch_Projects\\iCaRL-TheanoLasagne\\EMNIST_Compare_Results\\'
        # draw_bias_difference(compare_dir, fv_index=2, task_num=20)

        # compare_dir = 'D:\\Projects\\Projects\\pytorch_Projects\\iCaRL-TheanoLasagne\\EMNIST_Compare_Results\\'
        # draw_patience_difference(compare_dir, fv_index=1, task_num=20)


        # fv_list = [1, 2, 3, 4]
        #
        # for fv_i in fv_list:
        #     compare_dir = 'D:\\Projects\\Projects\\pytorch_Projects\\iCaRL-TheanoLasagne\\train_test_result\\'
        #     draw_diff_samples_acc_result(compare_dir, fv_index=fv_i, task_num=task_num, dataset_nums=3, drop='withoutdropout')