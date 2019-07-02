import numpy as np
from pyod.models.knn import KNN
from pyod.models.sod import SOD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import h5py

def stringToArray(string):
    temp_list = string.split(';')
    temp_list.pop()
    code_len = len(temp_list)
    array = np.zeros(shape=code_len,dtype=np.float32)
    for i in range(code_len):
        array[i] = np.float32(temp_list[i])
    return array

def arrayToString(class_id,array):
    size = array.size
    outliner_num = np.sum(array)
    this_string = "%s(%s):"%(class_id,outliner_num)
    for i in range(size):
        this_code = array[i]
        this_string = '%s%s,'%(this_string,this_code)
    return this_string

def get_specificClass_FV(lines,choose_class_id):
    FV_list = []
    for line in lines:
        class_id = line.split(',')[0]
        class_id = int(class_id.split(':')[1])
        if class_id == choose_class_id:
            code_string = line.split(',')[3]
            code_string = code_string.split(':')[1]
            code_array = stringToArray(code_string)
            FV_list.append(code_array)
    return FV_list

def IsolationForest_Method(data, outliers_fraction=0.001, seed=0):
    # 训练孤立森林模型
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)

    # 返回1表示正常值，-1表示异常值
    temp = model.predict(data)
    index_list = np.where(temp==-1)
    print("Outlier number: {}".format(len(index_list[0])))
    return index_list[0]

def get_dataset_outliner_using_KNN(train_FV_array,test_FV_array, outliers_fraction=0.05):
    # train kNN detector
    clf_name = 'KNN'
    np.random.seed(0)
    clf = KNN(contamination=outliers_fraction, n_neighbors=5)
    clf.fit(train_FV_array)
    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores
    # get the prediction on the test data
    y_test_pred = clf.predict(test_FV_array)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(test_FV_array)  # outlier scores
    return y_train_pred,y_train_scores,y_test_pred,y_test_scores

def get_dataset_outliner_using_SOD(train_FV_array,test_FV_array):
    # train kNN detector
    clf_name = 'SOD'
    clf = SOD(contamination=0.3, n_neighbors=20)
    clf.fit(train_FV_array)
    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores
    # get the prediction on the test data
    y_test_pred = clf.predict(test_FV_array)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(test_FV_array)  # outlier scores
    return y_train_pred,y_train_scores,y_test_pred,y_test_scores

def load_RawMNIST(class_num=7):
    import torchvision
    train_data = torchvision.datasets.MNIST('D:\Projects\Projects\pytorch_Projects\data',
                                            train=True, download=False)
    temp = np.array(train_data.train_labels==class_num)
    temp_list = []
    current_train_index_list = []
    for id, item in enumerate(temp):
        if item==0:
            pass
        else:
            temp_list.append(item)
            current_train_index_list.append(id)
    current_train_data = train_data.train_data[current_train_index_list]
    test_data = torchvision.datasets.MNIST('D:\Projects\Projects\pytorch_Projects\data',
                                           train=False, download=False)
    temp3 = np.array(test_data.test_labels == class_num)
    temp_list = []
    current_test_index_list = []
    for id, item in enumerate(temp3):
        if item == 0:
            pass
        else:
            temp_list.append(item)
            current_test_index_list.append(id)
    current_test_data = test_data.test_data[current_test_index_list]
    print(current_train_data.shape, current_test_data.shape)

    return current_train_data.numpy(), current_test_data.numpy(), current_train_index_list, current_test_index_list

def get_FV_outliner(num_class,FV_trainset_file,FV_testset_file,trainset_inliner_record_txt,testset_inliner_record_txt, outliers_fraction=0.05):
    trainset_lines = FV_trainset_file.readlines()
    testset_lines = FV_testset_file.readlines()
    trainset_inliner_sample_map = {}
    testset_inliner_sample_map = {}
    for class_id in range(num_class):
        this_class_trainset, this_class_testset, _, _ = load_RawMNIST(class_id)
        this_class_trainset = this_class_trainset.reshape(this_class_trainset.shape[0], -1) / 255.0
        this_class_testset = this_class_testset.reshape(this_class_testset.shape[0], -1) / 255.0
        # print(this_class_trainset[0].sum(), this_class_trainset[1].sum())
        # this_class_trainset = get_specificClass_FV(trainset_lines, class_id)
        # this_class_testset = get_specificClass_FV(testset_lines,class_id)
        this_class_trainset_array = np.array(this_class_trainset)
        this_class_testset_array = np.array(this_class_testset)

        train_pred, train_scores, test_pred, test_scores=get_dataset_outliner_using_KNN(this_class_trainset_array, this_class_testset_array, outliers_fraction=outliers_fraction)
        # train_pred, train_scores, test_pred, test_scores = get_dataset_outliner_using_SOD(this_class_trainset_array,
        #                                                                                   this_class_testset_array)
        trainset_inliner_sample_map[class_id] = train_pred
        testset_inliner_sample_map[class_id] = test_pred
        train_pred_string = arrayToString(class_id,train_pred)
        test_pred_string = arrayToString(class_id,test_pred)
        trainset_inliner_record_txt.write('%s\n'%train_pred_string)
        testset_inliner_record_txt.write('%s\n'%test_pred_string)
    return trainset_inliner_sample_map,testset_inliner_sample_map

def tSNE_dataset(all_train_data, seed=0):
    import numpy as np
    from sklearn.manifold import TSNE
    np.random.seed(seed=seed)
    tsne_train = TSNE(n_components=2)  # , init='pca', random_state=0)
    tsne_train_result = tsne_train.fit_transform(all_train_data)
    x_min, x_max = np.min(tsne_train_result, 0), np.max(tsne_train_result, 0)
    tsne_train_result = (tsne_train_result - x_min) / (x_max - x_min)


    plt.scatter(tsne_train_result[:, 0], tsne_train_result[:, 1])
    # plt.show()


def tSNE_Outlier_result(outliers_fraction=0.01):
    outlier_trainset_record_file = 'data\\0611_data\\new_FV_trainset_outlier_KNN_f({}).txt'.format(outliers_fraction)
    outlier_testset_record_file = 'data\\0611_data\\new_FV_testset_outlier_KNN_f({}).txt'.format(outliers_fraction)
    np.random.seed(0)
    f_train = open(outlier_trainset_record_file, 'r')
    train_records = f_train.readlines()
    f_train.close()
    f_test = open(outlier_testset_record_file, 'r')
    test_records = f_test.readlines()
    f_test.close()

    f_records_list = [train_records]#, test_records]
    # f_records_list = [test_records]
    isTest = 0
    for records in f_records_list:
        for each_class_str in records:  # got 10 classes tSNE result
            current_class = int(each_class_str.split('(')[0])
            current_class_outlier_list = []
            for data in each_class_str.split(':')[-1].split(',')[:-1]:
                current_class_outlier_list.append(int(data))
            current_class_outlier_list = np.array(current_class_outlier_list)
            if isTest==0:
                current_raw_mnist, _, _, _ = load_RawMNIST(current_class)
            else:
                _, current_raw_mnist, _, _ = load_RawMNIST(current_class)

            outlier_index_list = np.where(current_class_outlier_list==1)
            inlier_index_list = np.where(current_class_outlier_list==0)

            # saving mnist_dataset 2 hd5f file


            ############## tSNE result ##################
            # current_raw_mnist = current_raw_mnist.reshape(current_raw_mnist.shape[0], -1) / 255.0
            # tsne_train = TSNE(n_components=2)  # , init='pca', random_state=0)
            # tsne_train_result = tsne_train.fit_transform(current_raw_mnist)
            # x_min, x_max = np.min(tsne_train_result, 0), np.max(tsne_train_result, 0)
            # tsne_train_result = (tsne_train_result - x_min) / (x_max - x_min)
            #
            # plt.scatter(tsne_train_result[inlier_index_list[0], 0], tsne_train_result[inlier_index_list[0], 1], label='inliers')
            # plt.hold
            # plt.scatter(tsne_train_result[outlier_index_list[0], 0], tsne_train_result[outlier_index_list[0], 1],
            #             label='outliers', color='r')
            # plt.legend()
            # plt.subplots_adjust(top=0.95, bottom=0.05, left=0.02, right=0.98, hspace=0, wspace=0)

            ### plotting Outliers raw image
            img_num = 0
            show_image_num=0
            one_image_show_mnist_num = 100
            max_image_show = int(len(outlier_index_list[0]) / one_image_show_mnist_num)
            all_img_metric = np.zeros((28*int(np.sqrt(one_image_show_mnist_num)), 28*int(np.sqrt(one_image_show_mnist_num))))
            for id in outlier_index_list[0]:
                temp_image = current_raw_mnist[id, :, :]
                column = int(img_num % int(np.sqrt(one_image_show_mnist_num)))
                row = int(img_num / int(np.sqrt(one_image_show_mnist_num)))

                all_img_metric[28*row: 28*(row+1), 28*column: 28*(column+1)] = temp_image

                img_num+=1
                if (img_num%one_image_show_mnist_num==0 and not img_num==0) or (show_image_num==max_image_show and img_num==int(len(outlier_index_list[0]) % one_image_show_mnist_num)):
                    plt.imshow(all_img_metric, cmap=plt.cm.gray)
                    plt.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0, wspace=0)
                    plt.axis('off')
                    if isTest==1:
                        plt.savefig('result/Outliers_mnist/testset/({})class_testset_imgNum{}_outliers.png'.format(current_class,
                                                                                                    show_image_num),
                                    dpi=300)

                    else:
                        plt.savefig('result/Outliers_mnist/trainset/({})class_trainset_imgNum{}_outliers.png'.format(current_class, show_image_num), dpi=300)
                    # plt.show()
                    all_img_metric = np.zeros((28*int(np.sqrt(one_image_show_mnist_num)), 28*int(np.sqrt(one_image_show_mnist_num))))
                    show_image_num+=1
                    img_num=0
                if show_image_num>max_image_show:
                    break
            print(all_img_metric.shape)

            # break
        break
        # isTest+=1

def Saving_MNIST(outliers_fraction=0.1):
    outlier_trainset_record_file = 'data\\0611_data\\new_FV_trainset_outlier_KNN_f({}).txt'.format(outliers_fraction)
    outlier_testset_record_file = 'data\\0611_data\\new_FV_testset_outlier_KNN_f({}).txt'.format(outliers_fraction)
    np.random.seed(0)
    f_train = open(outlier_trainset_record_file, 'r')
    train_records = f_train.readlines()
    f_train.close()
    f_test = open(outlier_testset_record_file, 'r')
    test_records = f_test.readlines()
    f_test.close()

    f_records_list = [train_records, test_records]
    isTest = 0
    for records in f_records_list:
        for i, each_class_str in enumerate(records):  # got 10 classes tSNE result
            current_class = int(each_class_str.split('(')[0])
            current_class_outlier_list = []
            for data in each_class_str.split(':')[-1].split(',')[:-1]:
                current_class_outlier_list.append(int(data))
            current_class_outlier_list = np.array(current_class_outlier_list)
            if isTest == 0:
                current_raw_mnist, _, _, _ = load_RawMNIST(current_class)
            else:
                _, current_raw_mnist, _, _ = load_RawMNIST(current_class)

            outlier_index_list = np.where(current_class_outlier_list == 1)
            inlier_index_list = np.where(current_class_outlier_list == 0)

            if i==0:
                all_data = current_raw_mnist[inlier_index_list[0]]
                all_data_labels = current_class * np.ones(len(inlier_index_list[0]))
            else:
                all_data = np.concatenate((all_data, current_raw_mnist[inlier_index_list[0]]), axis=0)
                all_data_labels = np.concatenate((all_data_labels, current_class * np.ones(len(inlier_index_list[0]))), axis=0)

        # saving data
        path = 'data/0611_data/Outliers_Kicked_MNIST/'
        if isTest==0:
            f = h5py.File(path + 'Outliers_Kicked_MNIST_train_f({}).h5'.format(outliers_fraction), 'w')
            f.create_dataset('X_train', data=all_data)
            f.create_dataset('Y_train', data=all_data_labels)
            f.close()
        else:
            f = h5py.File(path + 'Outliers_Kicked_MNIST_test_f({}).h5'.format(outliers_fraction), 'w')
            f.create_dataset('X_test', data=all_data)
            f.create_dataset('Y_test', data=all_data_labels)
            f.close()

        isTest+=1

if __name__ == '__main__':
    # num_class = 10
    # outliers_fraction = 0.1
    # FV_trainset_path = 'data\\0611_data\\new_FV_trainset.txt'
    # FV_testset_path = 'data\\0611_data\\new_FV_testset.txt'
    # FV_trainset_file = open('%s'%FV_trainset_path,'r')
    # FV_testset_file = open('%s'%FV_testset_path,'r')
    # trainset_inliner_record_path = 'data\\0611_data\\new_FV_trainset_outlier.txt'
    # trainset_inliner_record_txt = open('%s'%trainset_inliner_record_path,'w')
    # testset_inliner_record_path = 'data\\0611_data\\new_FV_testset_outlier.txt'
    # testset_inliner_record_txt = open('%s'%testset_inliner_record_path,'w')
    # trainset_inliner_sample_map, testset_inliner_sample_map = get_FV_outliner(num_class, FV_trainset_file, FV_testset_file,
    #                         trainset_inliner_record_txt, testset_inliner_record_txt, outliers_fraction=outliers_fraction)


    outliers_fraction=0.1
    # tSNE_Outlier_result(outliers_fraction=outliers_fraction)
    #
    Saving_MNIST(outliers_fraction=outliers_fraction)