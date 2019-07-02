import utils_emnist
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import colorsys

import time
from sklearn import datasets
from sklearn.manifold import TSNE

def random_colors(N, bright=True, seed=0):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    # np.random.seed(seed)
    random.seed(seed)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    #
    # hsv = [(random.random(), random.random(), random.random()) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def fewshot_compare2testset_tSNE(fv_index=1, n_com=2, task_num=10, fewshot_fv_dir='FV-train50test50', mode=False, SEED=0):

    np.random.seed(SEED)
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

    from sklearn.manifold import TSNE
    temp_list = fewshot_fv_dir.split('train')[-1].split('test')
    temp = int(temp_list[0])
    temp2 = int(temp_list[-1])
    nb_val = temp
    batch_size = nb_val
    im_per_cls = batch_size
    p = float(temp2) / float(temp)

    if task_num==20:
        data, n_inputs = utils_emnist.load_data(0, im_per_cls, task_num, fv_index, choose_fv_dir='Remaked_EMNIST/FV-train50test50', p=p, val_mode=False)
    elif task_num==100:
        data, n_inputs = utils_emnist.load_data(0, im_per_cls, task_num, fv_index,
                                                choose_fv_dir='Remaked_CIFAR100/FV-train10test10', p=p, val_mode=False)
    else:
        data, n_inputs = utils_emnist.load_data(0, im_per_cls, task_num, fv_index, choose_fv_dir=fewshot_fv_dir, p=p, val_mode=False)
    X_train_total = data['X_train']
    Y_train_total = data['Y_train']
    X_valid_total = data['X_test']
    Y_valid_total = data['Y_test']

    each_class_trainsample_num = int(X_train_total.shape[0] / task_num)
    each_class_testsample_num = int(p * each_class_trainsample_num)
    trainset_data_metric = np.zeros([task_num, each_class_trainsample_num, X_train_total.shape[1]])
    testset_data_metric = np.zeros([task_num, each_class_testsample_num, X_valid_total.shape[1]])
    trainset_data_y_metric = np.zeros([task_num, each_class_trainsample_num])
    testset_data_y_metric = np.zeros([task_num, each_class_testsample_num])


    order = np.arange(task_num)
    # get each class train data and test data, then store to metric
    for iteration in range(task_num):
        indices_train_10 = np.array(
            [i in order[range(iteration, (iteration + 1))] for i in Y_train_total])
        indices_test_10 = np.array(
            [i in order[range(iteration, (iteration + 1))] for i in Y_valid_total])
        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        trainset_data_y_metric[iteration] = iteration + np.ones([each_class_trainsample_num])
        testset_data_y_metric[iteration] = iteration + np.ones([each_class_testsample_num])

        trainset_data_metric[iteration] = X_train
        testset_data_metric[iteration] = X_valid

    original_color_list = ['black', 'gray', 'lightcoral', 'red', 'blue', 'saddlebrown', 'peru', 'darkorange', 'gold', 'olive',
                  'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'slategray', 'midnightblue', 'indigo',
                  'deeppink', 'crimson']


    # random.seed(SEED)
    # random.shuffle(decay_list)
    if task_num == 100:
        color_list = original_color_list
        cmap = plt.get_cmap('gist_ncar')
        decay_list = np.linspace(0, 1, task_num - 20)
        for k in range(task_num - 20):
            color_list.append(cmap(decay_list[k]))
        # color_list = ['black', 'gray', 'lightcoral', 'red', 'blue', 'saddlebrown', 'peru', 'darkorange', 'gold',
        #               'olive',
        #               'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'slategray', 'midnightblue',
        #               'indigo', 'deeppink', 'crimson',
        #               'k', 'dimgray', 'dimgrey', 'grey',
        #               'sliver', 'lightgrey', 'gainsboro', 'rosybrown',
        #               'indianred', 'brown', 'firebrick', 'maroon',
        #               ]

    else:
        color_list = original_color_list


    # bright_color_list = random_colors(50, bright=True, seed=SEED)
    # dark_color_list = random_colors(50, bright=False, seed=SEED)
    # color_list = []
    # for color_idx in range(50):
    #     color_list.append(bright_color_list[color_idx])
    #     color_list.append(dark_color_list[color_idx])


    for index in range(task_num):
        if index==0:
            all_train_data = trainset_data_metric[index]
            all_train_data_y = trainset_data_y_metric[index]
            all_test_data = testset_data_metric[index]
            all_test_data_y = testset_data_y_metric[index]

            all_data = np.concatenate((trainset_data_metric[index], testset_data_metric[index]))
            all_data_y = np.concatenate((trainset_data_y_metric[index], testset_data_y_metric[index]))

        else:
            all_train_data = np.concatenate((all_train_data, trainset_data_metric[index]))
            all_train_data_y = np.concatenate((all_train_data_y, trainset_data_y_metric[index]))
            all_test_data = np.concatenate((all_test_data, testset_data_metric[index]))
            all_test_data_y = np.concatenate((all_test_data_y, testset_data_y_metric[index]))

            all_data = np.concatenate((all_data, np.concatenate((trainset_data_metric[index], testset_data_metric[index]))))
            all_data_y = np.concatenate((all_data_y, np.concatenate((trainset_data_y_metric[index], testset_data_y_metric[index]))))




    tsne_train = TSNE(n_components=n_com)#, init='pca', random_state=0)
    tsne_train_result = tsne_train.fit_transform(all_train_data)
    tsne_test = TSNE(n_components=n_com)#, init='pca', random_state=0)
    tsne_test_result = tsne_test.fit_transform(all_test_data)

    tsne_all = TSNE(n_components=n_com)
    tsne_all_result = tsne_all.fit_transform(all_data)

    # # load fewshot samples indices
    # train30test50_indices, train10test50_indices = get_samples_indices_from_original_dataset(task_num=task_num, fv_index=fv_index)


    x_min, x_max = np.min(tsne_train_result, 0), np.max(tsne_train_result, 0)
    tsne_train_result = (tsne_train_result - x_min) / (x_max - x_min)
    x_min, x_max = np.min(tsne_test_result, 0), np.max(tsne_test_result, 0)
    tsne_test_result = (tsne_test_result - x_min) / (x_max - x_min)
    x_min, x_max = np.min(tsne_all_result, 0), np.max(tsne_all_result, 0)
    tsne_all_result = (tsne_all_result - x_min) / (x_max - x_min)


    ################  trian50test50(test50)  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (i + 1) * (each_class_trainsample_num + each_class_testsample_num), 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (i + 1) * (each_class_trainsample_num + each_class_testsample_num), 1],
                    s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
        plt.hold

    # if task_num==20:
    #     plt.legend(loc='upper right', fontsize='xx-small')
    # else:
    #     plt.legend(loc='upper right', fontsize='small')
    plt.ylim([-0.03, 1.03])
    plt.xlim([-0.03, 1.03])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)

    plt.title('{}_{}_FV{}'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(testset)', fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}.png'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(testset)', fv_index), dpi=300)
    plt.close()

    ################  trian50test50(train50)  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
                        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 1],
                    s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(temp), (i + 1)))
        plt.hold


    # if task_num == 20:
    #     plt.legend(loc='upper right', fontsize='xx-small')
    # else:
    #     plt.legend(loc='upper right', fontsize='small')
    plt.ylim([-0.03, 1.03])
    plt.xlim([-0.03, 1.03])
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)

    plt.title('{}_{}_FV{}'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(trainset)', fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}.png'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(trainset)', fv_index), dpi=300)
    plt.close()

    ################  trian50test50  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        # plt.scatter(tsne_train_result[i * each_class_trainsample_num: (i+1) * each_class_trainsample_num, 0],
        #             tsne_train_result[i * each_class_trainsample_num: (i+1) * each_class_trainsample_num, 1],
        #             c=color_list[i], marker='o', label='train{}_class:{}'.format(int(temp), (i + 1)))
        # plt.hold
        #
        # plt.scatter(tsne_test_result[i * each_class_testsample_num: (i + 1) * each_class_testsample_num, 0],
        #         tsne_train_result[i * each_class_testsample_num: (i + 1) * each_class_testsample_num, 1],
        #         c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
        # plt.hold

        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
                        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 1],
                    s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(temp), (i + 1)))
        plt.hold

        plt.scatter(tsne_all_result[
                    i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
                                                                                                               i + 1) * (
                                                                                                               each_class_trainsample_num + each_class_testsample_num),
                    0],
                    tsne_all_result[
                    i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
                                                                                                               i + 1) * (
                                                                                                               each_class_trainsample_num + each_class_testsample_num),
                    1],
                    s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
        plt.hold

    # if task_num == 20:
    #     plt.legend(loc='upper right', fontsize='xx-small')
    # else:
    #     plt.legend(loc='upper right', fontsize='small')
    plt.ylim([-0.03, 1.03])
    plt.xlim([-0.03, 1.03])
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)

    plt.title('{}_{}_FV{}'.format(dataset, fewshot_fv_dir.split('-')[-1], fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}.png'.format(dataset, fewshot_fv_dir.split('-')[-1], fv_index), dpi=300)
    plt.close()
    #
    # ################  trian10test50  ###################
    # # need train30 correspond indices; and train10 correspond indices
    # train30sample = 30
    # train10sample = 10
    # plt.figure(figsize=(10, 5))
    # for i in range(task_num):
    #     temp_indices = train10test50_indices[i*train10sample: i*train10sample+train10sample]
    #     plt.scatter(tsne_all_result[temp_indices + i*50, 0],
    #                 tsne_all_result[temp_indices + i*50, 1],
    #                 s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(10), (i + 1)))
    #     plt.hold
    #
    #     plt.scatter(tsne_all_result[
    #                 i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
    #                                                                                                            i + 1) * (
    #                                                                                                            each_class_trainsample_num + each_class_testsample_num),
    #                 0],
    #                 tsne_all_result[
    #                 i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
    #                                                                                                            i + 1) * (
    #                                                                                                            each_class_trainsample_num + each_class_testsample_num),
    #                 1],
    #                 s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
    #     plt.hold
    #
    # # if task_num == 20:
    # #     plt.legend(loc='upper right', fontsize='xx-small')
    # # else:
    # #     plt.legend(loc='upper right', fontsize='small')
    # plt.ylim([-0.03, 1.03])
    # plt.xlim([-0.03, 1.03])
    # plt.xticks([])
    # plt.yticks([])
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)
    #
    # if task_num==100:
    #     plt.title('{}_{}_FV{}'.format(dataset, 'train3test10', fv_index))
    #     # plt.show()
    #     plt.savefig('{}_{}_FV{}.png'.format(dataset, 'train3test10', fv_index), dpi=300)
    # else:
    #     plt.title('{}_{}_FV{}'.format(dataset, 'train10test50', fv_index))
    #     # plt.show()
    #     plt.savefig('{}_{}_FV{}.png'.format(dataset, 'train10test50', fv_index), dpi=300)
    # plt.close()
    #
    # ################  trian30test50  ###################
    # plt.figure(figsize=(10, 5))
    # for i in range(task_num):
    #     temp_indices = train30test50_indices[i * train30sample: i * train30sample + train30sample]
    #     plt.scatter(tsne_all_result[temp_indices + i * 50, 0],
    #                 tsne_all_result[temp_indices + i * 50, 1],
    #                 s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(30), (i + 1)))
    #     plt.hold
    #
    #     plt.scatter(tsne_all_result[
    #                 i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
    #                                                                                                                i + 1) * (
    #                                                                                                                each_class_trainsample_num + each_class_testsample_num),
    #                 0],
    #                 tsne_all_result[
    #                 i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
    #                                                                                                                i + 1) * (
    #                                                                                                                each_class_trainsample_num + each_class_testsample_num),
    #                 1],
    #                 s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
    #     plt.hold
    #
    # # if task_num == 20:
    # #     plt.legend(loc='upper right', fontsize='xx-small')
    # # else:
    # #     plt.legend(loc='upper right', fontsize='small')
    #
    # plt.ylim([-0.03, 1.03])
    # plt.xlim([-0.03, 1.03])
    # plt.xticks([])
    # plt.yticks([])
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)
    #
    # if task_num==100:
    #     plt.title('{}_{}_FV{}'.format(dataset, 'train6test10', fv_index))
    #     # plt.show()
    #     plt.savefig('{}_{}_FV{}.png'.format(dataset, 'train6test10', fv_index), dpi=300)
    # else:
    #     plt.title('{}_{}_FV{}'.format(dataset, 'train30test50', fv_index))
    #     # plt.show()
    #     plt.savefig('{}_{}_FV{}.png'.format(dataset, 'train30test50', fv_index), dpi=300)
    # plt.close()

def NewFV20190424_MNIST_EMNIST_tSNE(fv_index=1, n_com=2, task_num=10, fewshot_fv_dir='NEW-FV-train50test50', mode=False, SEED=0):

    np.random.seed(SEED)
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

    from sklearn.manifold import TSNE
    temp_list = fewshot_fv_dir.split('train')[-1].split('test')
    temp = int(temp_list[0])
    temp2 = int(temp_list[-1])
    nb_val = temp
    batch_size = nb_val
    im_per_cls = batch_size
    p = float(temp2) / float(temp)

    if task_num==20:
        data, n_inputs = utils_emnist.load_patterns(0, im_per_cls, task_num, fv_index, choose_fv_dir='New_Remaked_EMNIST/FV-train50test50', p=p, val_mode=False)
    else:
        data, n_inputs = utils_emnist.load_patterns(0, im_per_cls, task_num, fv_index, choose_fv_dir=fewshot_fv_dir, p=p, val_mode=False)
    X_train_total = data['X_train']
    Y_train_total = data['Y_train']
    X_valid_total = data['X_test']
    Y_valid_total = data['Y_test']

    each_class_trainsample_num = int(X_train_total.shape[0] / task_num)
    each_class_testsample_num = int(p * each_class_trainsample_num)
    trainset_data_metric = np.zeros([task_num, each_class_trainsample_num, X_train_total.shape[1]])
    testset_data_metric = np.zeros([task_num, each_class_testsample_num, X_valid_total.shape[1]])
    trainset_data_y_metric = np.zeros([task_num, each_class_trainsample_num])
    testset_data_y_metric = np.zeros([task_num, each_class_testsample_num])


    order = np.arange(task_num)
    # get each class train data and test data, then store to metric
    for iteration in range(task_num):
        indices_train_10 = np.array(
            [i in order[range(iteration, (iteration + 1))] for i in Y_train_total])
        indices_test_10 = np.array(
            [i in order[range(iteration, (iteration + 1))] for i in Y_valid_total])
        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        trainset_data_y_metric[iteration] = iteration + np.ones([each_class_trainsample_num])
        testset_data_y_metric[iteration] = iteration + np.ones([each_class_testsample_num])

        trainset_data_metric[iteration] = X_train
        testset_data_metric[iteration] = X_valid

    color_list = ['black', 'gray', 'lightcoral', 'red', 'blue', 'saddlebrown', 'peru', 'darkorange', 'gold', 'olive',
                  'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'crimson', 'midnightblue', 'indigo', 'deeppink', 'crimson']


    for index in range(task_num):
        if index==0:
            all_train_data = trainset_data_metric[index]
            all_train_data_y = trainset_data_y_metric[index]
            all_test_data = testset_data_metric[index]
            all_test_data_y = testset_data_y_metric[index]

            all_data = np.concatenate((trainset_data_metric[index], testset_data_metric[index]))
            all_data_y = np.concatenate((trainset_data_y_metric[index], testset_data_y_metric[index]))

        else:
            all_train_data = np.concatenate((all_train_data, trainset_data_metric[index]))
            all_train_data_y = np.concatenate((all_train_data_y, trainset_data_y_metric[index]))
            all_test_data = np.concatenate((all_test_data, testset_data_metric[index]))
            all_test_data_y = np.concatenate((all_test_data_y, testset_data_y_metric[index]))

            all_data = np.concatenate((all_data, np.concatenate((trainset_data_metric[index], testset_data_metric[index]))))
            all_data_y = np.concatenate((all_data_y, np.concatenate((trainset_data_y_metric[index], testset_data_y_metric[index]))))




    tsne_train = TSNE(n_components=n_com)#, init='pca', random_state=0)
    tsne_train_result = tsne_train.fit_transform(all_train_data)
    tsne_test = TSNE(n_components=n_com)#, init='pca', random_state=0)
    tsne_test_result = tsne_test.fit_transform(all_test_data)

    tsne_all = TSNE(n_components=n_com)
    tsne_all_result = tsne_all.fit_transform(all_data)

    # # load fewshot samples indices
    # train30test50_indices, train10test50_indices = get_samples_indices_from_original_dataset(task_num=task_num, fv_index=fv_index)


    x_min, x_max = np.min(tsne_train_result, 0), np.max(tsne_train_result, 0)
    tsne_train_result = (tsne_train_result - x_min) / (x_max - x_min)
    x_min, x_max = np.min(tsne_test_result, 0), np.max(tsne_test_result, 0)
    tsne_test_result = (tsne_test_result - x_min) / (x_max - x_min)
    x_min, x_max = np.min(tsne_all_result, 0), np.max(tsne_all_result, 0)
    tsne_all_result = (tsne_all_result - x_min) / (x_max - x_min)


    ################  trian50test50(test50)  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (i + 1) * (each_class_trainsample_num + each_class_testsample_num), 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (i + 1) * (each_class_trainsample_num + each_class_testsample_num), 1],
                    s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
        plt.hold

    if task_num==20:
        plt.legend(loc='upper right', fontsize='xx-small')
    else:
        plt.legend(loc='upper right', fontsize='small')
    plt.ylim([0, 1])
    plt.xlim([0, 1.26])
    plt.xticks([])
    plt.yticks([])
    plt.title('{}_{}_FV{}-Patterns'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(testset)', fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}-Patterns.png'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(testset)', fv_index), dpi=300)
    plt.close()

    ################  trian50test50(train50)  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
                        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 1],
                    s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(temp), (i + 1)))
        plt.hold


    if task_num == 20:
        plt.legend(loc='upper right', fontsize='xx-small')
    else:
        plt.legend(loc='upper right', fontsize='small')
    plt.ylim([0, 1])
    plt.xlim([0, 1.26])
    plt.xticks([])
    plt.yticks([])
    plt.title('{}_{}_FV{}-Patterns'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(trainset)', fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}-Patterns.png'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(trainset)', fv_index), dpi=300)
    plt.close()

    ################  trian50test50  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        # plt.scatter(tsne_train_result[i * each_class_trainsample_num: (i+1) * each_class_trainsample_num, 0],
        #             tsne_train_result[i * each_class_trainsample_num: (i+1) * each_class_trainsample_num, 1],
        #             c=color_list[i], marker='o', label='train{}_class:{}'.format(int(temp), (i + 1)))
        # plt.hold
        #
        # plt.scatter(tsne_test_result[i * each_class_testsample_num: (i + 1) * each_class_testsample_num, 0],
        #         tsne_train_result[i * each_class_testsample_num: (i + 1) * each_class_testsample_num, 1],
        #         c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
        # plt.hold

        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
                        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 1],
                    s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(temp), (i + 1)))
        plt.hold

        plt.scatter(tsne_all_result[
                    i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
                                                                                                               i + 1) * (
                                                                                                               each_class_trainsample_num + each_class_testsample_num),
                    0],
                    tsne_all_result[
                    i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
                                                                                                               i + 1) * (
                                                                                                               each_class_trainsample_num + each_class_testsample_num),
                    1],
                    s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
        plt.hold

    if task_num == 20:
        plt.legend(loc='upper right', fontsize='xx-small')
    else:
        plt.legend(loc='upper right', fontsize='small')
    plt.ylim([0, 1])
    plt.xlim([0, 1.26])
    plt.xticks([])
    plt.yticks([])
    plt.title('{}_{}_FV{}-Patterns'.format(dataset, fewshot_fv_dir.split('-')[-1], fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}-Patterns.png'.format(dataset, fewshot_fv_dir.split('-')[-1], fv_index), dpi=300)
    plt.close()

def get_samples_indices_from_original_dataset(task_num=10, fv_index=1):
    if task_num == 10:
        dataset = 'MNIST-FV'
        orig_hidden_dim = 50
        SNN_STRUCTURE='15-50-10'

    elif task_num == 20:
        dataset = 'EMNIST-FV'
        orig_hidden_dim = 67
        SNN_STRUCTURE = '20-67-20'

    elif task_num == 100:
        dataset = 'CIFAR100-FV'
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'


    train50test50_data = None
    train30test50_data = None
    train10test50_data = None

    if task_num==20:
        data_dir = 'data/Remaked_EMNIST/'
    else:
        data_dir = 'data/'
    if not task_num==100:
        for dirs in os.listdir(data_dir):
            if dirs=='FV-train50test50':
                dataset_path = data_dir + dirs + '/'
                for dataset_dir in os.listdir(dataset_path):
                    if dataset_dir==dataset:
                        files_path = dataset_path + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'trainset' in files and 'FV{}'.format(fv_index) in files:
                                f = open(files_path + files)
                                train50test50_data = f.readlines()
                                f.close()
            elif dirs=='FV-train30test50':
                dataset_path = data_dir + dirs + '/'
                for dataset_dir in os.listdir(dataset_path):
                    if dataset_dir == dataset:
                        files_path = dataset_path + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'trainset' in files and 'FV{}'.format(fv_index) in files:
                                f = open(files_path + files)
                                train30test50_data = f.readlines()
                                f.close()
            elif dirs=='FV-train10test50':
                dataset_path = data_dir + dirs + '/'
                for dataset_dir in os.listdir(dataset_path):
                    if dataset_dir == dataset:
                        files_path = dataset_path + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'trainset' in files and 'FV{}'.format(fv_index) in files:
                                f = open(files_path + files)
                                train10test50_data = f.readlines()
                                f.close()
        index_list_train30test50 = []
        index_list_train10test50 = []
        for item_train30test50 in train30test50_data:
            for index, item_train50test50 in enumerate(train50test50_data):
                if item_train30test50==item_train50test50:
                    index_list_train30test50.append(index)

        for item_train10test50 in train10test50_data:
            for index, item_train50test50 in enumerate(train50test50_data):
                if item_train10test50==item_train50test50:
                    index_list_train10test50.append(index)

        iCaRL_Load_index = np.arange(0, 1000)

        # if task_num==10:
        #     iCaRL_Load_index = np.arange(0, 1000)
        # elif task_num==20:
        #     iCaRL_Load_index = []
        #     for i in range(task_num):
        #         for index, item_train50test50 in enumerate(train50test50_data):
        #             if item_train50test50.split(',')[0]=='class:{}'.format(i):
        #                 iCaRL_Load_index.append(index)
        #     iCaRL_Load_index = np.array(iCaRL_Load_index)



        index_list_train30test50 = iCaRL_Load_index[index_list_train30test50]
        index_list_train10test50 = iCaRL_Load_index[index_list_train10test50]
        return index_list_train30test50, index_list_train10test50

    else:
        for dirs in os.listdir(data_dir):
            if dirs == 'FV-train10test10':
                dataset_path = data_dir + dirs + '/'
                for dataset_dir in os.listdir(dataset_path):
                    if dataset_dir == dataset:
                        files_path = dataset_path + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'trainset' in files and 'FV{}'.format(fv_index) in files:
                                f = open(files_path + files)
                                train50test50_data = f.readlines()
                                f.close()
            elif dirs == 'FV-train6test10':
                dataset_path = data_dir + dirs + '/'
                for dataset_dir in os.listdir(dataset_path):
                    if dataset_dir == dataset:
                        files_path = dataset_path + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'trainset' in files and 'FV{}'.format(fv_index) in files:
                                f = open(files_path + files)
                                train30test50_data = f.readlines()
                                f.close()
            elif dirs == 'FV-train3test10':
                dataset_path = data_dir + dirs + '/'
                for dataset_dir in os.listdir(dataset_path):
                    if dataset_dir == dataset:
                        files_path = dataset_path + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            if 'trainset' in files and 'FV{}'.format(fv_index) in files:
                                f = open(files_path + files)
                                train10test50_data = f.readlines()
                                f.close()
        index_list_train30test50 = []
        index_list_train10test50 = []
        for item_train30test50 in train30test50_data:
            for index, item_train50test50 in enumerate(train50test50_data):
                if item_train30test50 == item_train50test50:
                    index_list_train30test50.append(index)

        for item_train10test50 in train10test50_data:
            for index, item_train50test50 in enumerate(train50test50_data):
                if item_train10test50 == item_train50test50:
                    index_list_train10test50.append(index)

        iCaRL_Load_index = np.arange(0, 1000)


        index_list_train30test50 = iCaRL_Load_index[index_list_train30test50]
        index_list_train10test50 = iCaRL_Load_index[index_list_train10test50]
        return index_list_train30test50, index_list_train10test50

def remake_EMNIST_FV_dataset(dirs, task_num=20, new=False):


    if task_num == 10:
        dataset = 'MNIST-FV'
        orig_hidden_dim = 50
        SNN_STRUCTURE='15-50-10'

    elif task_num == 20:
        dataset = 'EMNIST-FV'
        orig_hidden_dim = 67
        SNN_STRUCTURE = '20-67-20'

    elif task_num == 100:
        dataset = 'CIFAR100-FV'
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'

    if new:
        save_dir = dirs + 'New_Remaked_EMNIST/'
    else:
        if task_num==100:
            save_dir = dirs + 'Remaked_CIFAR100/'
        elif task_num==20:
            save_dir = dirs + 'Remaked_EMNIST/'

    if new:
    # load train50test50
        trainset_list = ['NEW-FV-train50test50']
    else:
        if task_num==20:
            trainset_list = ['FV-train50test50', 'FV-train30test50', 'FV-train10test50']
        elif task_num==100:
            trainset_list = ['FV-train10test10', 'FV-train6test10', 'FV-train3test10']

    for traintest_dir in os.listdir(dirs):
        for traintest in trainset_list:
            if traintest in traintest_dir and not '.rar' in traintest_dir:
                dataset_dirs = dirs + traintest_dir + '/'
                for dataset_dir in os.listdir(dataset_dirs):
                    if dataset_dir==dataset:
                        files_path = dataset_dirs + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            f = open(files_path + files)
                            data = f.readlines()
                            f.close()

                            # Remake data
                            f_save = open(save_dir + '{}/{}/'.format(traintest, dataset) + files, 'w')
                            for idx in range(task_num):
                                for item in data:
                                    if item.split(',')[0]=='class:{}'.format(idx):
                                        f_save.write(item)
                            f_save.close()

def remake_Double_MNIST_FV_dataset(dirs, task_num=10):

    if task_num == 10:
        dataset = 'MNIST-FV'
        orig_hidden_dim = 50
        SNN_STRUCTURE='15-50-10'

    elif task_num == 20:
        dataset = 'EMNIST-FV'
        orig_hidden_dim = 67
        SNN_STRUCTURE = '20-67-20'

    elif task_num == 100:
        dataset = 'CIFAR100-FV'
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'

    trainset_list = ['FV-train50test50']
    save_dir = dirs + 'New_Tripled_MNIST/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for traintest_dir in os.listdir(dirs):
        for traintest in trainset_list:
            if traintest in traintest_dir and not '.rar' in traintest_dir:
                dataset_dirs = dirs + traintest_dir + '/'
                for dataset_dir in os.listdir(dataset_dirs):
                    if dataset_dir==dataset:
                        files_path = dataset_dirs + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            f = open(files_path + files)
                            data = f.readlines()
                            f.close()
                            # Remake data
                            f_save = open(save_dir + files, 'w')
                            for idx in range(task_num):
                                for item in data:
                                    if item.split(',')[0]=='class:{}'.format(idx):
                                        title_str = item.split('code:')[0] + 'code:'
                                        data_str = item.split('code:')[-1]
                                        f_save.write(title_str)
                                        total_length = len(data_str.split(';'))
                                        for each_data in data_str.split(';'):
                                            if each_data=='\n':
                                                f_save.write(each_data)
                                            else:
                                                f_save.write(each_data + ';')
                                                f_save.write(each_data + ';')
                                                f_save.write(each_data + ';')

                            f_save.close()

def extract_subset_EMNIST_FV_dataset_then_remake(dirs, task_num=20, need_task_num=10, new=False):
    if task_num == 10:
        dataset = 'MNIST-FV'
        orig_hidden_dim = 50
        SNN_STRUCTURE='15-50-10'
    elif task_num == 20:
        dataset = 'EMNIST-FV'
        orig_hidden_dim = 67
        SNN_STRUCTURE = '20-67-20'
    elif task_num == 100:
        dataset = 'CIFAR100-FV'
        orig_hidden_dim = 167
        SNN_STRUCTURE = '50-167-100'

    if new:
        save_dir = dirs + 'New_Remaked_EMNIST/'
    else:
        if task_num==100:
            save_dir = dirs + 'Remaked_CIFAR100/'
        elif task_num==20:
            save_dir = dirs + 'New_Remaked_EMNIST/'

    if new:
    # load train50test50
        trainset_list = ['NEW-FV-train50test50']
        tmp_name = 'patterns'
    else:
        tmp_name = 'FV'
        if task_num==20:
            trainset_list = ['FV-train50test50']
        elif task_num==100:
            trainset_list = ['FV-train10test10']

    for traintest_dir in os.listdir(dirs):
        for traintest in trainset_list:
            if traintest in traintest_dir and not '.rar' in traintest_dir:
                dataset_dirs = dirs + traintest_dir + '/'
                for dataset_dir in os.listdir(dataset_dirs):
                    if dataset_dir==dataset:
                        files_path = dataset_dirs + dataset_dir + '/'
                        for files in os.listdir(files_path):
                            f = open(files_path + files)
                            data = f.readlines()
                            f.close()

                            # Remake data
                            temp_path = save_dir + '{}/{}-extract_{}classes-{}/'.format(traintest, dataset, need_task_num, tmp_name)
                            if not os.path.exists(temp_path):
                                os.mkdir(temp_path)
                            f_save = open(save_dir + '{}/{}-extract_{}classes-{}/'.format(traintest, dataset, need_task_num, tmp_name) + files, 'w')
                            for idx in range(need_task_num):
                                for item in data:
                                    if new:
                                        if int(item.split('-')[0].split('_')[-1])==idx:
                                            f_save.write(item)
                                    else:
                                        if item.split(',')[0]=='class:{}'.format(idx):
                                            f_save.write(item)
                            f_save.close()

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main_tSNE(traindata, trainlabel, fv_index=1, task_num=10):
    # data, label, n_samples, n_features = get_data()
    # load FV data

    print('Computing t-SNE embedding')

    data = traindata
    label = trainlabel
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    result = tsne.fit_transform(data)

    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)
    for i in range(result.shape[0]):
        plt.plot(result[i, 0], result[i, 1], 'o', color=plt.cm.Set1(label[i] / 10.), label='cls:{}'.format(label[i]))
        # plt.text(result[i, 0], result[i, 1], str(label[i]),
        #          color=plt.cm.Set1(label[i] / 10.),
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.hold
    plt.hold

    plt.title('FV{}'.format(fv_index))
    plt.legend()
    plt.show()

def FV_tSNE(fv_index=1, n_com=2, task_num=10, fewshot_fv_dir='FV-train50test50', mode=False, SEED=0):

    np.random.seed(SEED)
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

    from sklearn.manifold import TSNE

    nb_val = 50
    batch_size = nb_val
    im_per_cls = batch_size
    p = 1.0

    if task_num==20:
        data, n_inputs = utils_emnist.load_data(0, im_per_cls, task_num, fv_index, choose_fv_dir='Remaked_EMNIST/FV-train50test50', p=p, val_mode=False)
    elif task_num==100:
        data, n_inputs = utils_emnist.load_data(0, im_per_cls, task_num, fv_index,
                                                choose_fv_dir='Remaked_CIFAR100/FV-train10test10', p=p, val_mode=False)
    else:
        data, n_inputs = utils_emnist.load_data(0, im_per_cls, task_num, fv_index, choose_fv_dir=fewshot_fv_dir, p=p, val_mode=False)
    X_train_total = data['X_train']
    Y_train_total = data['Y_train']
    X_valid_total = data['X_test']
    Y_valid_total = data['Y_test']

    each_class_trainsample_num = int(X_train_total.shape[0] / task_num)
    each_class_testsample_num = int(p * each_class_trainsample_num)
    trainset_data_metric = np.zeros([task_num, each_class_trainsample_num, X_train_total.shape[1]])
    testset_data_metric = np.zeros([task_num, each_class_testsample_num, X_valid_total.shape[1]])
    trainset_data_y_metric = np.zeros([task_num, each_class_trainsample_num])
    testset_data_y_metric = np.zeros([task_num, each_class_testsample_num])


    order = np.arange(task_num)
    # get each class train data and test data, then store to metric
    for iteration in range(task_num):
        indices_train_10 = np.array(
            [i in order[range(iteration, (iteration + 1))] for i in Y_train_total])
        indices_test_10 = np.array(
            [i in order[range(iteration, (iteration + 1))] for i in Y_valid_total])
        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        trainset_data_y_metric[iteration] = iteration + np.ones([each_class_trainsample_num])
        testset_data_y_metric[iteration] = iteration + np.ones([each_class_testsample_num])

        trainset_data_metric[iteration] = X_train
        testset_data_metric[iteration] = X_valid

    original_color_list = ['black', 'gray', 'lightcoral', 'red', 'blue', 'saddlebrown', 'peru', 'darkorange', 'gold', 'olive',
                  'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'slategray', 'midnightblue', 'indigo',
                  'deeppink', 'crimson']


    # random.seed(SEED)
    # random.shuffle(decay_list)
    if task_num == 100:
        color_list = original_color_list
        cmap = plt.get_cmap('gist_ncar')
        decay_list = np.linspace(0, 1, task_num - 20)
        for k in range(task_num - 20):
            color_list.append(cmap(decay_list[k]))
        # color_list = ['black', 'gray', 'lightcoral', 'red', 'blue', 'saddlebrown', 'peru', 'darkorange', 'gold',
        #               'olive',
        #               'yellowgreen', 'lawngreen', 'palegreen', 'cyan', 'dodgerblue', 'slategray', 'midnightblue',
        #               'indigo', 'deeppink', 'crimson',
        #               'k', 'dimgray', 'dimgrey', 'grey',
        #               'sliver', 'lightgrey', 'gainsboro', 'rosybrown',
        #               'indianred', 'brown', 'firebrick', 'maroon',
        #               ]

    else:
        color_list = original_color_list


    # bright_color_list = random_colors(50, bright=True, seed=SEED)
    # dark_color_list = random_colors(50, bright=False, seed=SEED)
    # color_list = []
    # for color_idx in range(50):
    #     color_list.append(bright_color_list[color_idx])
    #     color_list.append(dark_color_list[color_idx])


    for index in range(task_num):
        if index==0:
            all_train_data = trainset_data_metric[index]
            all_train_data_y = trainset_data_y_metric[index]
            all_test_data = testset_data_metric[index]
            all_test_data_y = testset_data_y_metric[index]

            all_data = np.concatenate((trainset_data_metric[index], testset_data_metric[index]))
            all_data_y = np.concatenate((trainset_data_y_metric[index], testset_data_y_metric[index]))

        else:
            all_train_data = np.concatenate((all_train_data, trainset_data_metric[index]))
            all_train_data_y = np.concatenate((all_train_data_y, trainset_data_y_metric[index]))
            all_test_data = np.concatenate((all_test_data, testset_data_metric[index]))
            all_test_data_y = np.concatenate((all_test_data_y, testset_data_y_metric[index]))

            all_data = np.concatenate((all_data, np.concatenate((trainset_data_metric[index], testset_data_metric[index]))))
            all_data_y = np.concatenate((all_data_y, np.concatenate((trainset_data_y_metric[index], testset_data_y_metric[index]))))


    tsne_train = TSNE(n_components=n_com)#, init='pca', random_state=0)
    tsne_train_result = tsne_train.fit_transform(all_train_data)
    tsne_test = TSNE(n_components=n_com)#, init='pca', random_state=0)
    tsne_test_result = tsne_test.fit_transform(all_test_data)

    tsne_all = TSNE(n_components=n_com)
    tsne_all_result = tsne_all.fit_transform(all_data)

    # # load fewshot samples indices
    # train30test50_indices, train10test50_indices = get_samples_indices_from_original_dataset(task_num=task_num, fv_index=fv_index)


    x_min, x_max = np.min(tsne_train_result, 0), np.max(tsne_train_result, 0)
    tsne_train_result = (tsne_train_result - x_min) / (x_max - x_min)
    x_min, x_max = np.min(tsne_test_result, 0), np.max(tsne_test_result, 0)
    tsne_test_result = (tsne_test_result - x_min) / (x_max - x_min)
    x_min, x_max = np.min(tsne_all_result, 0), np.max(tsne_all_result, 0)
    tsne_all_result = (tsne_all_result - x_min) / (x_max - x_min)


    ################  trian50test50(test50)  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        plt.scatter(tsne_test_result[i * (each_class_testsample_num): (i + 1) * (each_class_testsample_num), 0],
                    tsne_test_result[i * (each_class_testsample_num): (i + 1) * (each_class_testsample_num), 1],
                    s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(50), (i + 1)))
        plt.hold

    # for i in range(task_num):
    #     plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (i + 1) * (each_class_trainsample_num + each_class_testsample_num), 0],
    #                 tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (i + 1) * (each_class_trainsample_num + each_class_testsample_num), 1],
    #                 s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(50), (i + 1)))
    #     plt.hold

    # if task_num==20:
    #     plt.legend(loc='upper right', fontsize='xx-small')
    # else:
    #     plt.legend(loc='upper right', fontsize='small')
    plt.ylim([-0.03, 1.03])
    plt.xlim([-0.03, 1.03])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)

    plt.title('{}_{}_FV{}'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(testset)', fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}.png'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(testset)', fv_index), dpi=300)
    plt.close()

    ################  trian50test50(train50)  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
                        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 1],
                    s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(50), (i + 1)))
        plt.hold


    # if task_num == 20:
    #     plt.legend(loc='upper right', fontsize='xx-small')
    # else:
    #     plt.legend(loc='upper right', fontsize='small')
    plt.ylim([-0.03, 1.03])
    plt.xlim([-0.03, 1.03])
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)

    plt.title('{}_{}_FV{}'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(trainset)', fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}.png'.format(dataset, fewshot_fv_dir.split('-')[-1] + '(trainset)', fv_index), dpi=300)
    plt.close()

    ################  trian50test50  ###################
    plt.figure(figsize=(10, 5))

    for i in range(task_num):
        # plt.scatter(tsne_train_result[i * each_class_trainsample_num: (i+1) * each_class_trainsample_num, 0],
        #             tsne_train_result[i * each_class_trainsample_num: (i+1) * each_class_trainsample_num, 1],
        #             c=color_list[i], marker='o', label='train{}_class:{}'.format(int(temp), (i + 1)))
        # plt.hold
        #
        # plt.scatter(tsne_test_result[i * each_class_testsample_num: (i + 1) * each_class_testsample_num, 0],
        #         tsne_train_result[i * each_class_testsample_num: (i + 1) * each_class_testsample_num, 1],
        #         c=color_list[i], marker='<', label='test{}_class:{}'.format(int(temp2), (i + 1)))
        # plt.hold

        plt.scatter(tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 0],
                    tsne_all_result[i * (each_class_trainsample_num + each_class_testsample_num): i * (
                        each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num, 1],
                    s=10, c=color_list[i], marker='o', label='train{}_class:{}'.format(int(50), (i + 1)))
        plt.hold

        plt.scatter(tsne_all_result[
                    i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
                                                                                                               i + 1) * (
                                                                                                               each_class_trainsample_num + each_class_testsample_num),
                    0],
                    tsne_all_result[
                    i * (each_class_trainsample_num + each_class_testsample_num) + each_class_trainsample_num: (
                                                                                                               i + 1) * (
                                                                                                               each_class_trainsample_num + each_class_testsample_num),
                    1],
                    s=10, c=color_list[i], marker='<', label='test{}_class:{}'.format(int(50), (i + 1)))
        plt.hold

    # if task_num == 20:
    #     plt.legend(loc='upper right', fontsize='xx-small')
    # else:
    #     plt.legend(loc='upper right', fontsize='small')
    plt.ylim([-0.03, 1.03])
    plt.xlim([-0.03, 1.03])
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)

    plt.title('{}_{}_FV{}'.format(dataset, fewshot_fv_dir.split('-')[-1], fv_index))
    # plt.show()
    plt.savefig('{}_{}_FV{}.png'.format(dataset, fewshot_fv_dir.split('-')[-1], fv_index), dpi=300)
    plt.close()


def main():
    data, label, n_samples, n_features = get_data()
    import torch
    import torch.nn.functional as F

    temp_data = torch.from_numpy(data)
    temp_data = F.sigmoid(temp_data)

    data = temp_data.numpy()


    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time.time() - t0))
    plt.show(fig)





if __name__ == '__main__':
    task_num = 10

    # dirs = 'data/'
    # remake_EMNIST_FV_dataset(dirs, task_num=task_num, new=False)

    # dir_list = ['FV-train50test50']  # MNIST and EMNIST20
    # # dir_list = ['FV-train10test10']  # CIFAR100
    # # # # dir_list = ['FV-train10test10', 'FV-train3test10', 'FV-train6test10']  # CIFAR100
    # # #
    # fv_list = [1, 2, 3, 4]
    # for fv in fv_list:
    #     for dirs in dir_list:
    #         fewshot_compare2testset_tSNE(task_num=task_num, fv_index=fv, fewshot_fv_dir=dirs, mode=True, SEED=0)


    dir_list = ['FVnew-train50test50-0619']# MNIST and EMNIST20
    FV_tSNE(1, fewshot_fv_dir=dir_list[0])

    # dir_list = ['FVnew-train50test50-0619']# MNIST and EMNIST20
    #
    # fv_list = [1]#, 2, 3, 4]
    # for fv in fv_list:
    #     for dirs in dir_list:
    #         NewFV20190424_MNIST_EMNIST_tSNE(task_num=task_num, fv_index=fv, fewshot_fv_dir=dirs, mode=True)

    # main()
    #
    # get_samples_indices_from_original_dataset(task_num, fv_index=3)

    # dirs = 'data/'
    # # dirs = 'data/NEW-FV-train50test50/'
    # extract_subset_EMNIST_FV_dataset_then_remake(dirs, task_num=task_num, need_task_num=10, new=True)


    # dirs = 'data/'
    # remake_Double_MNIST_FV_dataset(dirs)