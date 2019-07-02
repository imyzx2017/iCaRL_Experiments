import numpy as np
import os
import matplotlib
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import h5py

def getPatternFileName(class_id,phase,file_name_list,dataset_type):
    this_file_name = 'null'
    for file_name in file_name_list:
        temp_list = file_name.split('_0-%s_'%class_id)
        if len(temp_list) >=2:
            t_list = file_name.split('%s'%phase)
            if len(t_list)>=2:
                tt_list = file_name.split('%sDataset'%dataset_type)
                if len(tt_list)>=2:
                    this_file_name = file_name
                    break
    if this_file_name == 'null':
        raise AssertionError
    return this_file_name

def stringToArray(string):
    temp_list = string.split(';')
    temp_list.pop()
    code_len = len(temp_list)
    array = np.zeros(shape=code_len,dtype=np.float32)
    for i in range(code_len):
        array[i] = np.float32(temp_list[i])
    return array

def parse_lines(learningStep,num_sample_per_class,lines,map):
    for line in lines:
        class_id = line.split(',')[1]
        code_string = line.split(',')[2]
        code_array = stringToArray(code_string)
        if ('T%s-C%s'%(learningStep,class_id) in map.keys()) == False:
            map['T%s-C%s'%(learningStep,class_id)] = []
        map['T%s-C%s'%(learningStep,class_id)].append(code_array)
    return 0

def get_neurons_innerClass_stability(num_sample,num_neuron,firing_rate_list):
    firing_rate_array = np.array(firing_rate_list)
    assert firing_rate_array.shape == (num_sample,num_neuron)
    firing_rate_mean = np.average(firing_rate_array,0)
    assert firing_rate_mean.size == num_neuron
    firing_rate_mean = np.reshape(firing_rate_mean,[1,num_neuron])
    temp_array = np.sum(np.power(np.subtract(firing_rate_array,firing_rate_mean),2),0)
    innerClass_stability_array = 1.0/(num_sample-1)*temp_array
    innerClass_stability_array = np.sqrt(innerClass_stability_array)
    return innerClass_stability_array

def get_neurons_correlation_from_startTime_to_endTime(num_sample,num_neuron,s_FR_list,e_FR_list):
    s_FR_array = np.array(s_FR_list)
    e_FR_array = np.array(e_FR_list)
    assert s_FR_array.shape == e_FR_array.shape == (num_sample,num_neuron)
    s_FR_neuron_mean = np.average(s_FR_array,0)
    s_FR_neuron_mean = np.reshape(s_FR_neuron_mean,(1,num_neuron))
    e_FR_neuron_mean = np.average(e_FR_array,0)
    e_FR_neuron_mean = np.reshape(e_FR_neuron_mean,(1,num_neuron))
    s_FR_neuron_std = np.std(s_FR_array,0)
    e_FR_neuron_std = np.std(e_FR_array,0)
    ###
    s_FR_neuron_std[s_FR_neuron_std==0] = 1
    e_FR_neuron_std[e_FR_neuron_std == 0] = 1
    ###
    s_FR_subtract_mean = np.subtract(s_FR_array,s_FR_neuron_mean)
    e_FR_subtract_mean = np.subtract(e_FR_array,e_FR_neuron_mean)
    FR_neuron_cov = np.zeros(shape=s_FR_neuron_std.shape)
    for neuron_id in range(num_neuron):
        this_neuron_s_FR = s_FR_subtract_mean[:,neuron_id]
        this_neuron_e_FR = e_FR_subtract_mean[:,neuron_id]
        this_neuron_cov = np.abs(np.average(np.multiply(this_neuron_s_FR,this_neuron_e_FR)))
        FR_neuron_cov[neuron_id] = this_neuron_cov
    neurons_correlation_array = np.divide(FR_neuron_cov,np.multiply(s_FR_neuron_std,e_FR_neuron_std))
    return neurons_correlation_array

def get_neuronsFR_correlation_between_adjacentLearningStep(num_sample,num_neuron,s_FR_list,e_FR_list):
    s_FR_array = np.array(s_FR_list)
    e_FR_array = np.array(e_FR_list)
    assert s_FR_array.shape == e_FR_array.shape == (num_sample, num_neuron)
    s_e_FR_multiply = np.multiply(s_FR_array,e_FR_array)
    Expectation_s_e = np.average(s_e_FR_multiply,0)##E_s_e
    Expectation_s = np.average(s_FR_array,0)##E_s
    Expectation_e = np.average(e_FR_array,0)##E_e
    cov_s_e = np.subtract(Expectation_s_e,np.multiply(Expectation_s,Expectation_e))
    s_FR_neuron_std = np.std(s_FR_array, 0)
    e_FR_neuron_std = np.std(e_FR_array, 0)
    neuron_FR_correlation = np.zeros(shape=s_FR_neuron_std.shape)
    for neuron_id in range(num_neuron):
        this_s_std = s_FR_neuron_std[neuron_id]
        this_e_std = e_FR_neuron_std[neuron_id]
        if this_s_std == 0 or this_e_std == 0:
            neuron_FR_correlation[neuron_id] = 0.0
        else:
            neuron_FR_correlation[neuron_id] = np.abs(cov_s_e[neuron_id]/(this_s_std*this_e_std))
    return neuron_FR_correlation

def get_normFRPattern(FR_array,num_sample,num_neuron):
    FR_array_mode = np.sqrt(np.sum(np.power(FR_array, 2), 1))
    FR_array_mode = np.reshape(FR_array_mode, (num_sample, 1))
    FR_array_norm = np.divide(FR_array, FR_array_mode)
    return FR_array_norm

def get_sampleFRPattern_distance_between_adjacentLearningStep(num_sample,num_neuron,s_FR_list,e_FR_list,distance_method='Cos'):
    s_FR_array = np.array(s_FR_list)
    e_FR_array = np.array(e_FR_list)
    assert s_FR_array.shape == e_FR_array.shape == (num_sample, num_neuron)
    ##normalization each sample FR pattern to mode 1
    s_FR_array_norm = get_normFRPattern(s_FR_array,num_sample,num_neuron)
    e_FR_array_norm = get_normFRPattern(e_FR_array,num_sample,num_neuron)
    ##compute distance between some sample of adjacent learningStep
    if distance_method == 'EuroDis':
        distance_matrix = np.sum(np.power(np.subtract(s_FR_array_norm,e_FR_array_norm),2),1)
        distance_matrix = np.sqrt(distance_matrix)
        assert distance_matrix.size == num_sample
        all_sample_average_distance = np.average(distance_matrix)
        return all_sample_average_distance
    elif distance_method == "Cos":
        cos_matrix = np.sum(np.multiply(s_FR_array_norm,e_FR_array_norm),1)
        assert cos_matrix.size == num_sample
        all_sample_average_distance = np.average(cos_matrix)
        return all_sample_average_distance
    else:raise NotImplementedError

def get_each_neuron_max_firing_rate_during_learning(num_class,num_neuron,map,save_path):
    FR_pattern_list = []
    for learningStep in range(num_class):
        for class_id in range(0,learningStep+1):
            this_key_name = 'T%s-C%s'%(learningStep,class_id)
            assert (this_key_name in map.keys()) == True
            this_sample_list = map[this_key_name]
            FR_pattern_list.extend(this_sample_list)
    FR_pattern_array = np.array(FR_pattern_list)
    num_sample,num_n = FR_pattern_array.shape
    h5_file = h5py.File('%s/Testset_Firing_rate_matrix(%s-%s).hdf5'%(save_path,num_sample,num_n),'w')
    h5_file.create_dataset(name='data',data=FR_pattern_array)
    h5_file.close()
    neuron_FR_max = np.max(FR_pattern_array,0)
    assert neuron_FR_max.size == num_neuron
    return neuron_FR_max,FR_pattern_array

def get_all_firing_rate_pattern(num_class,num_neuron,map,save_path):
    FR_pattern_list = []
    for learningStep in range(num_class):
        for class_id in range(0, learningStep + 1):
            this_key_name = 'T%s-C%s' % (learningStep, class_id)
            assert (this_key_name in map.keys()) == True
            this_sample_list = map[this_key_name]
            FR_pattern_list.extend(this_sample_list)
    FR_pattern_array = np.array(FR_pattern_list)
    num_sample, num_n = FR_pattern_array.shape
    file_name = 'Testset_Firing_rate_pattern_matrix(%s-%s).hdf5'% (num_sample, num_n)
    h5_file = h5py.File('%s/%s'%(save_path,file_name) , 'w')
    h5_file.create_dataset(name='data', data=FR_pattern_array)
    h5_file.close()
    return file_name

def visualize_FR_distribution(save_path,h5_file_name,num_sample,num_neuron):
    h5_file = h5py.File('%s/%s' % (save_path,h5_file_name), 'r')
    FR_matrix = h5_file.get(name='data').value
    max_FR = np.max(FR_matrix)
    bins = int(np.ceil(max_FR))
    neuron_distribution_list = []
    for neuron_id in range(num_neuron):
        this_data = FR_matrix[:,neuron_id]
        this_neuron_hist,bin_edges =  np.histogram(a=this_data,bins=bins,normed=True,range=(0,np.ceil(max_FR)))
        neuron_distribution_list.append(this_neuron_hist)
    neuron_distribution_array = np.array(neuron_distribution_list)
    ratio = (num_neuron+0.0)/bins
    print(num_neuron,bins,ratio)
    fig, ax = plt.subplots()
    fig.set_size_inches(int(ratio * 10),8)
    ##
    neuron_distribution_array_T = np.transpose(neuron_distribution_array)
    im = ax.imshow(neuron_distribution_array_T, vmin=np.min(neuron_distribution_array), vmax=np.max(neuron_distribution_array))
    colorBar = fig.colorbar(mappable=im, ax=ax)
    ax.set_title('Neurons Firing rate distribution(maxFR=%s)'%max_FR)
    ax.set_ylabel('Firing Rate Range')
    ax.set_xlabel('Neuron Index')
    plt.savefig('%s/Neurons distribution.png' % (save_path))
    return neuron_distribution_array


if __name__ == '__main__':
    # METHODS = ['ANN Single', 'EWC', 'GEM', 'ANN Joint Train']
    METHODS = ['iCaRL']
    method = METHODS[0]
    num_class = 100
    num_sample_per_class = 10  # 50 for EMNIST and MNIST, 10 for CIFAR100
    num_neuron = 167
    dataset_name = 'CIFAR100' # 'MNIST', 'CIFAR100'
    # FV_level = 'FV1'
    # root_dir = 'D:\\vacation\continue-learn\continual-learn\\results\patterns'
    root_dir = 'D:\\Projects\Projects\pytorch_Projects\\0417_iCaRL_Result\\result\\patterns\\'
    for method in METHODS:
        for FV_level in ['FV1', 'FV2', 'FV3', 'FV4']:
            if method == 'iCaRL':
                current_path = '%s/iCaRL/%s/nodropout/hidden%d_1Layers/' % (root_dir, dataset_name, num_neuron)
            else:
                current_path = '%s/%s\%s/' % (root_dir, dataset_name.lower(), method.replace(' ', ''))
            Layer_name = 'H(%s)' % num_neuron
            dataset_path = '%s%s' % (current_path, FV_level)
            file_name_list = os.listdir(dataset_path)
            save_path = '%s%s-stability' % (current_path, FV_level)
            if os.path.exists(save_path) == False:
                os.mkdir(save_path)
            dataset_type = 'Test'
            build_map_flag = True
            innerClass_stability_flag = False
            specificClass_sampleFRPattern_stability_flag = True
            classAverageFRPatternDuringCL_stability_flag = True
            specificLearningStep_FRPattern_orthogonality_flag = True
            TestsetFR_in_each_learningStep_map = {}
            total_sample_present = 0
            if build_map_flag == True:
                for class_id in range(num_class):
                    test_txt_file_name = getPatternFileName(class_id=class_id, phase=Layer_name, file_name_list=file_name_list,
                                                        dataset_type=dataset_type)
                    test_txt_path = os.path.join(dataset_path, test_txt_file_name)
                    this_test_file = open(test_txt_path,'r')
                    lines = this_test_file.readlines()
                    parse_lines(class_id,num_sample_per_class,lines,TestsetFR_in_each_learningStep_map)
                print('building TestsetFR map done!!!!')
                TestFR_FR_pattern_file_name = get_all_firing_rate_pattern(num_class, num_neuron,TestsetFR_in_each_learningStep_map, save_path)
                ####
                WHOLE_FR_matrix = np.zeros(shape=(num_neuron * num_class, num_sample_per_class * num_class), dtype=np.float32)
                for class_id in range(num_class):
                    for learningStep in range(class_id, num_class):
                        this_key = 'T%s-C%s' % (learningStep, class_id)
                        this_FR_list = TestsetFR_in_each_learningStep_map[this_key]
                        this_FR_array = np.array(this_FR_list)
                        this_FR_array_T = np.transpose(this_FR_array)
                        WHOLE_FR_matrix[class_id * num_neuron:(class_id + 1) * num_neuron,
                        learningStep * num_sample_per_class:(learningStep + 1) * num_sample_per_class] = this_FR_array_T
                ##save FR pattern matrix as HDF5 file
                FR_file = h5py.File('%s/%s(%s)-%s_Neurons_Firing_rate_during_CL.hdf5' % (save_path, dataset_name, FV_level, Layer_name),'w')
                FR_file.create_dataset(name='data', data=WHOLE_FR_matrix)
                FR_file.close()
            ###
            if innerClass_stability_flag == True:
                innerClass_data_path = os.path.join(save_path,'innerClass_stability')
                if os.path.exists(innerClass_data_path)==False:
                    os.mkdir(innerClass_data_path)
                neurons_innerClass_stability_tensor = np.zeros(shape=(num_class,num_class,num_neuron),dtype=np.float32)
                for learningStep in range(num_class):
                    for learningClass_id in range(learningStep+1):
                        key_name = 'T%s-C%s'%(learningStep,learningClass_id)
                        this_state_sampleFR_list = TestsetFR_in_each_learningStep_map[key_name]
                        this_stability = get_neurons_innerClass_stability(num_sample=num_sample_per_class,num_neuron=num_neuron,firing_rate_list=this_state_sampleFR_list)
                        this_stability = np.reshape(this_stability,(1,1,this_stability.size))
                        neurons_innerClass_stability_tensor[learningStep,learningClass_id,:] = this_stability
                neurons_innerClass_s_h5_file = h5py.File(name='%s/neurons_innerClass_stability.hdf5'%(innerClass_data_path),mode='w')
                neurons_innerClass_s_h5_file.create_dataset(name='data',data=neurons_innerClass_stability_tensor)
                stability_min = np.min(neurons_innerClass_stability_tensor)
                stability_max = np.max(neurons_innerClass_stability_tensor)
                for neuron_id in range(num_neuron):
                    this_neuron_stability_mat = neurons_innerClass_stability_tensor[:,:,neuron_id]
                    fig, ax = plt.subplots()
                    fig.set_size_inches(30, 30)
                    im = ax.imshow(this_neuron_stability_mat,vmin=stability_min, vmax=stability_max)
                    colorBar = fig.colorbar(mappable=im, ax=ax)
                    ax.set_title('Neuron%s InnerClass Stability'%neuron_id)
                    ax.set_xlabel('Class Index')
                    ax.set_ylabel('LearningStep')
                    plt.savefig('%s/Neuron%s InnerClass Stability.png' % (innerClass_data_path, neuron_id))
            ###
            if specificClass_sampleFRPattern_stability_flag == True and build_map_flag == True:
                specificClass_sampleFRPattern_distance_map = {}
                specificClass_sampleFRPattern_distance_mat = np.zeros(shape=(num_class-1,num_class-1),dtype=np.float32)
                for class_id in range(num_class-1):
                    this_ClassSampleFRPattern_averageDistance_btwLearning = 0
                    present_time = int(num_class-class_id-1)
                    if (class_id in specificClass_sampleFRPattern_distance_map.keys()) ==False:
                        specificClass_sampleFRPattern_distance_map[class_id] = []
                    for learningStep in range(class_id,num_class-1):
                        s_time_key_name = 'T%s-C%s' % (learningStep,class_id)
                        e_time_key_name = 'T%s-C%s' % (learningStep+1,class_id)
                        s_FR_list = TestsetFR_in_each_learningStep_map[s_time_key_name]
                        e_FR_list = TestsetFR_in_each_learningStep_map[e_time_key_name]
                        this_state_sampleFR_average_distance = get_sampleFRPattern_distance_between_adjacentLearningStep(num_sample_per_class,\
                                                        num_neuron,s_FR_list,e_FR_list)
                        print('class id%s,%s-%s,%s' % (class_id, learningStep, learningStep + 1, this_state_sampleFR_average_distance))
                        specificClass_sampleFRPattern_distance_mat[class_id,learningStep] = this_state_sampleFR_average_distance
                        specificClass_sampleFRPattern_distance_map[class_id].append(this_state_sampleFR_average_distance)
                    print('Class%s present time %s'%(class_id,present_time))
                ###
                specificClass_sampleFRPattern_distance_matrix_h5_file = h5py.File(name='%s/%s(%s)-specificClass_sampleFRPattern_cosDistance_matrix.hdf5' % (save_path, dataset_name, FV_level),mode='w')
                specificClass_sampleFRPattern_distance_matrix_h5_file.create_dataset(name='data',data=specificClass_sampleFRPattern_distance_mat)
                specificClass_sampleFRPattern_distance_matrix_h5_file.close()
            ###
            if classAverageFRPatternDuringCL_stability_flag == True and build_map_flag == True:
                classAverageFRPatternDuringCL_matrix = np.zeros(shape=(num_class-1,num_class-1),dtype=np.float32)
                for class_id in range(0,num_class-1):
                    for learningStep in range(class_id,num_class-1):
                        s_time_key_name = 'T%s-C%s' % (learningStep,class_id)
                        e_time_key_name = 'T%s-C%s' % (learningStep+1,class_id)
                        s_FR_list = TestsetFR_in_each_learningStep_map[s_time_key_name]
                        s_FR_array = np.array(s_FR_list)
                        s_FR_norm_array = get_normFRPattern(s_FR_array,num_sample_per_class,num_neuron)
                        s_FR_norm_average = np.average(s_FR_norm_array,0)##(num_sample, num_neuron)
                        ###
                        e_FR_list = TestsetFR_in_each_learningStep_map[e_time_key_name]
                        e_FR_array = np.array(e_FR_list)
                        e_FR_norm_array = get_normFRPattern(e_FR_array,num_sample_per_class,num_neuron)
                        e_FR_norm_average = np.average(e_FR_norm_array,0)
                        ###
                        assert s_FR_norm_average.size == e_FR_norm_average.size == num_neuron
                        distance_btw_adjacentLearningStep = np.sqrt(np.sum(np.power(np.subtract(s_FR_norm_average,e_FR_norm_average),2)))
                        print('class id%s,%s-%s,%s' % (class_id, learningStep, learningStep + 1, distance_btw_adjacentLearningStep))
                        classAverageFRPatternDuringCL_matrix[class_id,learningStep] = distance_btw_adjacentLearningStep
                    classAverageFRPatternDuringCL_matrix_h5_file = h5py.File(
                        name='%s/%s(%s)-classAverageFRPatternDistanceDuringCL_matrix.hdf5' % (
                        save_path, dataset_name, FV_level), mode='w')
                    classAverageFRPatternDuringCL_matrix_h5_file.create_dataset(name='data',data=classAverageFRPatternDuringCL_matrix)
                    classAverageFRPatternDuringCL_matrix_h5_file.close()
            ###
            if specificLearningStep_FRPattern_orthogonality_flag == True and build_map_flag == True:
                specificLearningStep_FRPattern_orthogonality_matrix = np.zeros(shape=(num_class-1,2),dtype=np.float32)
                for learningStep in range(1,num_class):
                    this_learningStep_orthogonality_data_list = []
                    for pre_class_id in range(0,learningStep+1):
                        for post_class_id in range(pre_class_id,learningStep+1):
                            if pre_class_id == post_class_id:pass
                            else:
                                pre_class_key_name = 'T%s-C%s' % (learningStep, pre_class_id)
                                post_class_key_name = 'T%s-C%s' % (learningStep,post_class_id)
                                pre_class_FR_list = TestsetFR_in_each_learningStep_map[pre_class_key_name]
                                post_class_FR_list = TestsetFR_in_each_learningStep_map[post_class_key_name]
                                this_state_adjacentClass_sampleFR_average_orthogonality_degree = get_sampleFRPattern_distance_between_adjacentLearningStep(num_sample_per_class, \
                                    num_neuron, pre_class_FR_list, post_class_FR_list)
                                print('learningStep%s,%s-%s,%s' % (learningStep, pre_class_id, post_class_id, this_state_adjacentClass_sampleFR_average_orthogonality_degree))
                                this_learningStep_orthogonality_data_list.append(this_state_adjacentClass_sampleFR_average_orthogonality_degree)

                # specificLearningStep_FRPattern_orthogonality_h5_file = h5py.File(name='%s/%s(%s)-specificLearningStep_FRPattern_orthogonality_matrix.hdf5' % (
                #     save_path, dataset_name, FV_level), mode='w')
                # specificLearningStep_FRPattern_orthogonality_h5_file.create_dataset(name='data', data=specificLearningStep_FRPattern_orthogonality_matrix)
                # specificLearningStep_FRPattern_orthogonality_h5_file.close()
                    this_learningStep_orthogonality_data_array = np.array(this_learningStep_orthogonality_data_list)
                    this_mean = np.average(this_learningStep_orthogonality_data_array)
                    this_std = np.std(this_learningStep_orthogonality_data_array)
                    specificLearningStep_FRPattern_orthogonality_matrix[learningStep-1,0] = this_mean
                    specificLearningStep_FRPattern_orthogonality_matrix[learningStep-1,1] = this_std
                specificLearningStep_FRPattern_orthogonality_matrix_h5_file = h5py.File(
                    name='%s/%s(%s)-specificLearningStep_FRPattern_orthogonality_matrix.hdf5' % (
                    save_path, dataset_name, FV_level), mode='w')
                specificLearningStep_FRPattern_orthogonality_matrix_h5_file.create_dataset(name='data',
                                                                                     data=specificLearningStep_FRPattern_orthogonality_matrix)
                specificLearningStep_FRPattern_orthogonality_matrix_h5_file.close()
