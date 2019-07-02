# THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu1,floatX=float32,lib.cnmem=0.09' python
from __future__ import print_function
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import utils_emnist
import convert_matrixcsv2npz

######### Modifiable Settings ##########
val_mode_list = [False]     # True for training , False for validation
# val_mode_list = [True]     # True for training , False for validation
small_sample_learning_mode = False
fv_dataset_idx_list = [1]#,2,3,4]
# choose_fv_dir_list = ['MNIST10-NOISE(0.3)', 'MNIST10-NOISE(0.5)', 'MNIST10-NOISE(0.7)']     # for MNIST_NOISE

# choose_fv_dir_list = ['FV-train50test50']     # for MNIST and EMNIST
choose_fv_dir_list = ['FVnew-train50test50-cleaned']     # for MNIST and EMNIST
# choose_fv_dir_list = ['CMNIST-NEW-BINARY-FV1']
# choose_fv_dir_list = ['FV-train10test10']   # for CIFAR100


hidden_choose = [[15]]#, [20], [50], [67], [100], [200], [400], [800], [1600], [2400], [3200]]    # for EMNIST

# hidden_choose = [[2400], [3200]]    # for EMNIST
# hidden_choose = [[20, 50, 67, 100, 200, 400, 800, 1600]]

# hidden_choose = [[20], [50], [100], [200], [400], [800], [1600], [2400], [3200]]      # for MNIST
# hidden_choose = [[50], [100], [167], [400], [800], [1600]]
# hidden_choose = [[800, 800], [800, 800, 800]]
# hidden_choose = [[67, 67], [67, 67, 67]]
# hidden_choose = [[400, 400], [400, 400, 400]]

# hidden_choose = [[50], [50, 50], [50, 50, 50]]
# hidden_choose = [[15]]

dropout_choose = [None, 0.5]
nb_val = 25  # Validation samples per class: 5 for CIFAR, 25 for MNIST


# n = 20  # input feature size

nb_all_cls = 10   # total classes: 20 for EMNIST
nb_protos = 1  # Number of prototypes per class at the end: total protoset memory/ total number of classes
epoch_list = [5000]
epochs = 5000  # Total number of epochs
shuffle = False
lr_old = 0.045  # Initial learning rate 0.05for NoBinaryFV
lr_strat = [35000, 45000]  # Epochs where learning rate gets decreased
lr_factor = 5.  # Learning rate decrease factor
wght_decay = 0.00001  # Weight Decay
seed_list = [0]
seed = seed_list[0]

validation_error_nochanged_epoch_setting = 100
# seed = 0
########################################
nb_tasks = nb_all_cls
nb_cl = nb_all_cls / nb_tasks  # Classes per group



try:
    temp_list = choose_fv_dir_list[0].split('train')[-1].split('test')
    temp = int(temp_list[0])
    temp2 = int(temp_list[-1])
    p = float(temp2)/float(temp)
except:
    p = 1.0

if small_sample_learning_mode:
    nb_val = temp
    batch_size = nb_val
    im_per_cls = batch_size



# n_hidden = [400]
# using_dropout_p = None

np.random.seed(seed)  # Fix the random seed
# Launch the different dataset
for val_mode in val_mode_list:
    if val_mode:
        batch_size = 2 * nb_val  # Batch size
    else:
        batch_size = nb_val
    im_per_cls = batch_size
    for current_fv_dir in choose_fv_dir_list:
        for fv_dataset_idx in fv_dataset_idx_list:
            for n_hidden in hidden_choose:
                for using_dropout_p in dropout_choose:
                    predict_mat = np.zeros((nb_all_cls, nb_all_cls))
                    for epochs in epoch_list:
                        acc_list = []

                        # Load the dataset
                        print("Loading data...")
                        if val_mode:
                            data, n_inputs = utils_emnist.load_data(0, im_per_cls, nb_all_cls, fv_dataset_idx, choose_fv_dir=current_fv_dir, p=p, val_mode=False)
                        else:
                            # adding small_samples_learning
                            if small_sample_learning_mode:
                                data, n_inputs = utils_emnist.load_data(0, im_per_cls, nb_all_cls, fv_dataset_idx,
                                                                        choose_fv_dir=current_fv_dir, p=p, val_mode=False)
                            else:
                                data, n_inputs = utils_emnist.load_data2(nb_all_cls, fv_dataset_idx, val_mode=False)
                        X_train_total = data['X_train']
                        Y_train_total = data['Y_train']
                        # if nb_val != 0:
                        #     X_valid_total = data['X_valid']
                        #     Y_valid_total = data['Y_valid']
                        # else:
                        #     X_valid_total = data['X_test']
                        #     Y_valid_total = data['Y_test']
                        if val_mode:
                            X_valid_total = data['X_test']
                            Y_valid_total = data['Y_test']
                        else:
                            if small_sample_learning_mode:
                                X_valid_total = data['X_train']
                                Y_valid_total = data['Y_train']
                            else:
                                X_valid_total = data['X_valid']
                                Y_valid_total = data['Y_valid']
                        # X_valid_total = data2['X_test']
                        # Y_valid_total = data2['Y_test']



                        # Initialization
                        dictionary_size = batch_size   # deafualt im_per_cls - nb_val = 50
                        top1_acc_list_cumul = np.zeros((nb_tasks, 3))
                        top1_acc_list_ori = np.zeros((nb_tasks, 3))

                        # Select the order for the class learning
                        order = np.arange(nb_all_cls)

                        # Prepare Theano variables for inputs and targets
                        input_var = T.matrix('inputs')
                        target_var = T.matrix('targets')

                        # Create neural network model
                        print("Building model and compiling functions...")
                        [network, intermed] = utils_emnist.build_MutipleLayersNN(input_var, n_inputs, n_hidden=n_hidden, n_output=nb_all_cls, dropout_p=using_dropout_p)

                        prediction = lasagne.layers.get_output(network)
                        # pattern = lasagne.layers.get_output(intermed)
                        loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
                        loss = loss.mean()
                        all_layers = lasagne.layers.get_all_layers(network)
                        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * wght_decay
                        loss = loss + l2_penalty

                        # Create a training function
                        params = lasagne.layers.get_all_params(network, trainable=True)
                        lr = lr_old
                        sh_lr = theano.shared(lasagne.utils.floatX(lr))
                        updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr)#, momentum=0.9)
                        train_fn = theano.function([input_var, target_var], loss, updates=updates)
                        # func_cur_pred = theano.function([input_var], [prediction])
                        # func_cur_pattern = theano.function([input_var], [pattern])

                        # Create a validation/testing function
                        test_prediction = lasagne.layers.get_output(network, deterministic=True)
                        test_prediction_intermed = lasagne.layers.get_output(intermed, deterministic=True)
                        test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
                        test_loss = test_loss.mean()
                        val_fn = theano.function([input_var, target_var], [test_loss, test_prediction, test_prediction_intermed])

                        # Create a feature mapping function
                        pred_map = lasagne.layers.get_output(intermed, deterministic=True)
                        function_map = theano.function([input_var], [pred_map])

                        # Initialization of the variables for this run
                        X_valid_cumuls = []
                        X_protoset_cumuls = []
                        X_train_cumuls = []
                        Y_valid_cumuls = []
                        Y_protoset_cumuls = []
                        Y_train_cumuls = []
                        alpha_dr_herding = np.zeros((nb_tasks, dictionary_size, nb_cl), np.float32)

                        # The following contains all the training samples of the different classes
                        # because we want to compare our method with the theoretical case where all the training samples are stored
                        prototypes = np.zeros(
                            (nb_all_cls, dictionary_size, X_train_total.shape[1])) # prototype container for all train examples
                        for orde in range(nb_all_cls):
                            prototypes[orde, :, :] = X_train_total[np.where(Y_train_total == order[orde])]

                        stop_epoch_list = []


                        #################################
                        ##############
                        dataset = ''
                        if nb_all_cls == 100:
                            dataset = 'CIFAR100'
                        elif nb_all_cls == 20:
                            dataset = 'EMNIST20'
                        elif nb_all_cls == 10:
                            dataset = 'MNIST'

                        import os

                        str_n_hidden = ''
                        str_dropout = ''
                        for item in n_hidden:
                            str_n_hidden += str(item)
                            str_n_hidden += '-'

                        if using_dropout_p is None:
                            str_dropout = 'WithoutDropout'
                        else:
                            str_dropout = 'Dropout(%f)' % using_dropout_p

                        if small_sample_learning_mode:
                            if not os.path.exists('result/%s/' % current_fv_dir):
                                os.mkdir('result/%s/' % current_fv_dir)
                            save_dir = 'result/%s/%s_FV_MLP%s-%s%s/' % (current_fv_dir, dataset, n_inputs, str_n_hidden, nb_all_cls)
                        else:
                            if not os.path.exists('result/%s/' % current_fv_dir):
                                os.mkdir('result/%s/' % current_fv_dir)
                            save_dir = 'result/%s/%s_FV_MLP%s-%s%s/' % (current_fv_dir, dataset, n_inputs, str_n_hidden, nb_all_cls)
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        stop_epochs_save_path = save_dir + '%s_MLP%s-%s%s_FV%d_%s_epoch%d_rand%d_stopped-epochs.txt' % (
                            dataset, n_inputs, str_n_hidden, nb_all_cls, fv_dataset_idx, str_dropout, epochs, seed)
                        #########################################






                        for iteration in range(nb_tasks):
                            # Save data results at each increment
                            # np.save('top1_acc_list_cumul_icarl_cl' + str(nb_cl), top1_acc_list_cumul)
                            # np.save('top1_acc_list_ori_icarl_cl' + str(nb_cl), top1_acc_list_ori)

                            # Prepare the training data for the current batch of classes
                            actual_cl = order[range(iteration * nb_cl, (iteration + 1) * nb_cl)]  # classes of current task
                            indices_train_10 = np.array(
                                [i in order[range(iteration * nb_cl, (iteration + 1) * nb_cl)] for i in Y_train_total])
                            indices_test_10 = np.array(
                                [i in order[range(iteration * nb_cl, (iteration + 1) * nb_cl)] for i in Y_valid_total])
                            X_train = X_train_total[indices_train_10]
                            X_valid = X_valid_total[indices_test_10]
                            X_valid_cumuls.append(X_valid)
                            X_train_cumuls.append(X_train)
                            X_valid_cumul = np.concatenate(X_valid_cumuls)
                            X_train_cumul = np.concatenate(X_train_cumuls)
                            Y_train = Y_train_total[indices_train_10]
                            Y_valid = Y_valid_total[indices_test_10]
                            Y_valid_cumuls.append(Y_valid)
                            Y_train_cumuls.append(Y_train)
                            Y_valid_cumul = np.concatenate(Y_valid_cumuls)
                            Y_train_cumul = np.concatenate(Y_train_cumuls)

                            # Add the stored exemplars to the training data
                            if iteration == 0:
                                X_valid_ori = X_valid
                                Y_valid_ori = Y_valid
                            else:
                                X_protoset = np.concatenate(X_protoset_cumuls)
                                Y_protoset = np.concatenate(Y_protoset_cumuls)
                                X_train = np.concatenate((X_train, X_protoset), axis=0)
                                Y_train = np.concatenate((Y_train, Y_protoset))

                            # Launch the training loop
                            sh_lr.set_value(lasagne.utils.floatX(lr_old))
                            print("\n")
                            print('Batch of classes number {0} arrives ...'.format(iteration + 1))


                            # adding stop flag
                            valid_acc_list = []


                            if val_mode:
                                epochs_list = utils_emnist.get_stopped_epoch_list(stop_epochs_save_path)
                                epochs = int(epochs_list[iteration])
                            for epoch in range(epochs):

                                ######################## train on train_set #######################
                                # Shuffle training data
                                if shuffle:
                                    train_indices = np.arange(len(X_train))
                                    np.random.shuffle(train_indices)
                                    X_train = X_train[train_indices, :]
                                    Y_train = Y_train[train_indices]


                                # In each epoch, we do a full pass over the training data:
                                train_err = 0
                                train_batches = 0

                                My_err = 0


                                start_time = time.time()
                                for batch in utils_emnist.iterate_minibatches(X_train, Y_train, batch_size, shuffle=shuffle, augment=False):
                                    inputs, targets_prep = batch
                                    targets = np.zeros((inputs.shape[0], nb_all_cls), np.float32)
                                    targets[range(len(targets_prep)), targets_prep.astype('int32')] = 1.  # one hot
                                    old_train = train_err
                                    # cur_prediction = func_cur_pred(inputs)[0]
                                    # cur_pattern = func_cur_pattern(inputs)[0]
                                    if iteration == 0:
                                        train_err += train_fn(inputs, targets)

                                    # Distillation
                                    if iteration > 0:
                                        prediction_old = func_pred(inputs)[0]
                                        targets[:, np.array(order[range(0, iteration * nb_cl)])] = prediction_old[:, np.array(
                                            order[range(0, iteration * nb_cl)])]
                                        train_err += train_fn(inputs, targets)

                                    # if (train_batches % 10) == 1:
                                    #     print(train_err - old_train)

                                    train_batches += 1

                                # # Shuffle validation data
                                # validation_indices = np.arange(len(X_valid))
                                # np.random.shuffle(validation_indices)
                                # X_valid = X_valid[validation_indices, :]
                                # Y_valid = Y_valid[validation_indices]

                                ############################################
                                #
                                # Ref to Ian Goodfellow 2015 papers
                                #
                                ############################################
                                # setting stopped method: train until the validation set error has not improved in the last 100 epochs.

                                if not val_mode:
                                    valid_acc = 0
                                    valid_batches = 0

                                    for batch in utils_emnist.iterate_minibatches(X_valid, Y_valid, batch_size, shuffle=shuffle,
                                                                                  augment=False):
                                        valid_inputs, valid_targets_orig = batch
                                        valid_targets = np.zeros((valid_inputs.shape[0], nb_all_cls), np.float32)
                                        valid_targets[range(len(valid_targets_orig)), valid_targets_orig.astype('int32')] = 1.
                                        err, pred, pred_inter = val_fn(inputs, targets)
                                        pred_ranked = pred.argsort(axis=1).argsort(axis=1)

                                        for i in range(valid_inputs.shape[0]):
                                            valid_acc += np.float((pred_ranked[i, valid_targets_orig[i]] >= (nb_all_cls - 1))) / \
                                                           inputs.shape[0]

                                        valid_batches += 1
                                    valid_acc_list.append(valid_acc)
                                    stopflag = utils_emnist.StopTrainingByAccInValidset(valid_acc_list, fv_dataset_idx,
                                                                                        validation_error_nochanged_epoch_setting)
                                    if stopflag:
                                        stop_epoch_list.append(epoch)
                                        print('Validation Accuracy:{}'.format(valid_acc_list[-100:]))
                                        break




                                # And a full pass over the validation data:
                                val_err = 0
                                top5_acc = 0
                                top1_acc = 0
                                val_batches = 0
                                # Then we print the results for this epoch:
                                if (epoch + 1) % 10 == 0:
                                    for batch in utils_emnist.iterate_minibatches(X_valid, Y_valid, min(batch_size, len(X_valid)),
                                                                                  shuffle=shuffle):
                                        inputs, targets_prep = batch
                                        targets = np.zeros((inputs.shape[0], nb_all_cls), np.float32)
                                        targets[range(len(targets_prep)), targets_prep.astype('int32')] = 1.
                                        err, pred, pred_inter = val_fn(inputs, targets)  # loss, prediction and hidden pattern
                                        pred_ranked = pred.argsort(axis=1).argsort(axis=1)

                                        for i in range(inputs.shape[0]):
                                            top5_acc = top5_acc + np.float((pred_ranked[i, targets_prep[i]] >= (nb_all_cls - 5))) / \
                                                       inputs.shape[0]
                                            top1_acc = top1_acc + np.float((pred_ranked[i, targets_prep[i]] >= (nb_all_cls - 1))) / \
                                                       inputs.shape[0]

                                        val_err += err
                                        val_batches += 1
                                    print('class %d, epoch %d, train loss %.5f, test accuracy %.4f' %
                                      (iteration + 1, epoch + 1, train_err / train_batches, top1_acc / val_batches))



                                # if (iteration > 0) and (epoch % 200 == 0):
                                #     print('pause')
                                # print("Batch of classes {} out of {} batches".format(
                                #     iteration + 1, nb_tasks))
                                # print("Epoch {} of {} took {:.3f}s".format(
                                #     epoch + 1, epochs, time.time() - start_time))
                                # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                                # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                                # print("  top 1 accuracy:\t\t{:.2f} %".format(
                                #     top1_acc / val_batches * 100))
                                # print("  top 5 accuracy:\t\t{:.2f} %".format(
                                #     top5_acc / val_batches * 100))
                                # adjust learning rate
                                if (epoch + 1) in lr_strat:
                                    new_lr = sh_lr.get_value() * 1. / lr_factor
                                    print("New LR:" + str(new_lr))
                                    sh_lr.set_value(lasagne.utils.floatX(new_lr))
                            ####################################################

                            # all epochs of one task done!



                            # Duplicate current network to distillate info
                            if iteration == 0:
                                [network2, intermed2] = utils_emnist.build_MutipleLayersNN(input_var, n_inputs, n_hidden=n_hidden, n_output=nb_all_cls, dropout_p=using_dropout_p)
                                prediction_distil = lasagne.layers.get_output(network2, deterministic=True)
                                prediction_features = lasagne.layers.get_output(intermed2, deterministic=True)
                                func_pred = theano.function([input_var], [prediction_distil])
                                func_pred_feat = theano.function([input_var], [prediction_features])


                            params_values = lasagne.layers.get_all_param_values(network)
                            lasagne.layers.set_all_param_values(network2, params_values)

                            # Save the network
                            # np.savez('result/net_incr' + str(iteration + 1) + '_of_' + str(nb_tasks) + '.npz',
                            #          *lasagne.layers.get_all_param_values(network))
                            # np.savez('result/intermed_incr' + str(iteration + 1) + '_of_' + str(nb_tasks) + '.npz',
                            #          *lasagne.layers.get_all_param_values(intermed))

                            ### Exemplars
                            nb_protos_cl = int(np.ceil(nb_protos * nb_tasks * 1. / (iteration + 1)))
                            # Herding
                            print('Updating exemplar set...')
                            for iter_dico in range(nb_cl):
                                # Possible exemplars in the feature space and projected on the L2 sphere
                                # normalize hidden layer patterns
                                # pattern of classes in current task
                                mapped_prototypes = function_map(np.float32(prototypes[iteration * nb_cl + iter_dico]))
                                D = mapped_prototypes[0].T
                                D = D / np.linalg.norm(D, axis=0)  # length of each pattern vector = 1

                                # Herding procedure : ranking of the potential exemplars
                                mu = np.mean(D, axis=1)
                                alpha_dr_herding[iteration, :, iter_dico] = alpha_dr_herding[iteration, :, iter_dico] * 0
                                w_t = mu
                                iter_herding = 0
                                iter_herding_eff = 0
                                while not (np.sum(alpha_dr_herding[iteration, :, iter_dico] != 0) == min(nb_protos_cl, im_per_cls)) and iter_herding_eff<10000:
                                    temp1 = np.sum(alpha_dr_herding[iteration, :, iter_dico] != 0)
                                    temp2 = min(nb_protos_cl, im_per_cls)
                                    tmp_t = np.dot(w_t, D)
                                    ind_max = np.argmax(tmp_t)  # max cos, which d have smallest angle with w_t
                                    iter_herding_eff += 1
                                    if alpha_dr_herding[iteration, ind_max, iter_dico] == 0:
                                        alpha_dr_herding[iteration, ind_max, iter_dico] = 1 + iter_herding
                                        iter_herding += 1
                                    w_t = w_t + mu - D[:, ind_max]

                            # Prepare the protoset
                            X_protoset_cumuls = []
                            Y_protoset_cumuls = []


                            ####################################################
                            # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
                            print('Computing mean-of_exemplars and theoretical mean...')
                            # class_means = np.zeros((n_hidden, nb_all_cls, 2))
                            class_means = np.ones((n_hidden[-1], nb_all_cls, 2)) * -1
                            for iteration2 in range(iteration + 1):
                                for iter_dico in range(nb_cl):
                                    current_cl = order[range(iteration2 * nb_cl, (iteration2 + 1) * nb_cl)]

                                    # Collect data in the feature space for each class
                                    mapped_prototypes = function_map(np.float32(prototypes[iteration2 * nb_cl + iter_dico]))
                                    D = mapped_prototypes[0].T
                                    D = D / np.linalg.norm(D, axis=0)
                                    # Flipped version also
                                    # mapped_prototypes2 = function_map(np.float32(prototypes[iteration2 * nb_cl + iter_dico][:, ::-1]))
                                    # D2 = mapped_prototypes2[0].T
                                    # D2 = D2 / np.linalg.norm(D2, axis=0)

                                    # iCaRL
                                    alph = alpha_dr_herding[iteration2, :, iter_dico]
                                    alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.

                                    X_protoset_cumuls.append(prototypes[iteration2 * nb_cl + iter_dico, np.where(alph == 1)[0]])
                                    Y_protoset_cumuls.append(order[iteration2 * nb_cl + iter_dico] * np.ones(len(np.where(alph == 1)[0])))
                                    alph = alph / np.sum(alph)
                                    # class_means[:, current_cl[iter_dico], 0] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                                    class_means[:, current_cl[iter_dico], 0] = np.dot(D, alph)
                                    class_means[:, current_cl[iter_dico], 0] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 0])

                                    # Normal NCM
                                    # alph = np.ones(dictionary_size) / dictionary_size
                                    # class_means[:, current_cl[iter_dico], 1] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                                    # class_means[:, current_cl[iter_dico], 1] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 1])

                            #####################################################
                            #
                            # Saving pattern after training on each group
                            #
                            #####################################################
                            if val_mode:
                                utils_emnist.saving_patterns_inFVdatasets(iteration, X_train_cumul, X_valid_cumul, dataset, fv_dataset_idx, func=function_map, n_hidden=n_hidden, batch_size=batch_size, using_dropout=using_dropout_p, small_samples_learning=small_sample_learning_mode)

                            # np.save('cl_means', class_means)
                            task_acc_list = []
                            for t, X_valid in enumerate(X_valid_cumuls):
                                print('Computing task %d accuracy...' % t)
                                task_acc, pm = utils_emnist.accuracy_measure(X_valid, Y_valid_cumuls[t], class_means, val_fn, iteration, 0, 'cumul of', im_per_cls,nb_all_cls)
                                if iteration == (nb_all_cls-1):
                                    predict_mat += pm
                                task_acc_list.append(task_acc)
                            acc_list.append(task_acc_list)
                            ####################################################



                            # Calculate validation error of model on the first nb_cl classes:
                            # print('Computing accuracy on the original batch of classes...')
                            # top1_acc_list_ori = utils_emnist.accuracy_measure(X_valid_ori, Y_valid_ori, class_means, val_fn,
                            #                                                     top1_acc_list_ori, iteration, 0, 'original', im_per_cls, nb_all_cls)
                            #
                            # # Calculate validation error of model on the cumul of classes:
                            # print('Computing cumulative accuracy...')
                            # top1_acc_list_cumul = utils_emnist.accuracy_measure(X_valid_cumul, Y_valid_cumul, class_means, val_fn,
                            #                                                       top1_acc_list_cumul, iteration, 0, 'cumul of', im_per_cls, nb_all_cls)
                            # task done!


                        ############################################
                        #
                        # Get int pred metric from test_set
                        #
                        ############################################



                        def save_result(result_a, fname=None):
                            import csv
                            with open(fname, 'wb') as f:
                                writer = csv.writer(f)
                                for line in result_a:
                                    try:
                                        str_line = [str(x) for x in line]
                                    except:
                                        str_line = line
                                    writer.writerow(str_line)

                        print(acc_list)



                        save_path = save_dir + '%s_MLP%s-%s%s_FV%d_%s_epoch%d_rand%d_accuracy.csv' % (dataset, n_inputs, str_n_hidden, nb_all_cls, fv_dataset_idx, str_dropout, epochs, seed)
                        if val_mode:
                            save_path = save_dir + '%s_MLP%s-%s%s_FV%d_%s_epoch%d_rand%d_accuracy.csv' % (dataset, n_inputs, str_n_hidden, nb_all_cls, fv_dataset_idx, str_dropout, epochs, seed)
                            save_result(acc_list, save_path)

                            # convert_matrixcsv2npz.load_csv2npy(save_dir, task_num=nb_all_cls, fv_index=fv_dataset_idx)



                            # np.savez(save_dir + '%s_MLP20-%s20_FV%d_%s_epoch%d_rand%d_accuracy_net_incr' % (dataset, str_n_hidden, fv_dataset_idx, str_dropout, epochs, seed) + '.npz',
                            #          *lasagne.layers.get_all_param_values(network))
                            # np.savez(save_dir + '%s_MLP20-%s20_FV%d_%s_epoch%d_rand%d_accuracy_intermed' % (dataset, str_n_hidden, fv_dataset_idx, str_dropout, epochs, seed) + '.npz',
                            #          *lasagne.layers.get_all_param_values(intermed))
                            np.save(save_dir + '%s_MLP20-%s20_FV%d_%s_epoch%d_rand%d_accuracy_predictmat' % (dataset, str_n_hidden, fv_dataset_idx, str_dropout, epochs, seed) + '.npy', predict_mat.T)

                        if not val_mode:
                            np.savetxt(stop_epochs_save_path, np.array(stop_epoch_list))


                # finished on one FV dataset
                if val_mode:
                    convert_matrixcsv2npz.load_csv2npy(save_dir, task_num=nb_all_cls, n_hidden=n_hidden, fv_index=fv_dataset_idx)

            # Final save of the data
            # np.save('result/top1_acc_list_cumul_icarl_cl' + str(nb_cl), top1_acc_list_cumul)
            # np.save('result/top1_acc_list_ori_icarl_cl' + str(nb_cl), top1_acc_list_ori)