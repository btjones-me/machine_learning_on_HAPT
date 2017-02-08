#####################################################################

# Example : load HAPT data set only
# basic illustrative python script

# For use with test / training datasets : HAPT-data-set-DU

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import plot_sklearn_report as skl_report
import matplotlib.pyplot as pyplot


########### Define classes

classes = {'WALKING' : 1, 'WALKING_UPSTAIRS' : 2, 'WALKING_DOWNSTAIRS' : 3, 'SITTING' : 4, 'STANDING' : 5, 'LAYING' : 6,'STAND_TO_SIT' : 7, 'SIT_TO_STAND' : 8, 'SIT_TO_LIE' : 9, 'LIE_TO_SIT' : 10, 'STAND_TO_LIE' : 11, 'LIE_TO_STAND' :12}
inv_classes = {v: k for k, v in classes.items()}

print(classes)
# print(inv_classes)

########### Define global variables

training_attributes = 0
training_class_labels = 0
testing_attributes = 0
testing_class_labels = 0

training_nn_outputs = 0 ##For nn only
testing_nn_outputs = 0 ##for nn only ##potentially not used

########### Load Data Set

path_to_data = "..//HAPT-data-set-DU" # edit this

# Training data - as currenrtly split
def read_data():
    attribute_list = []
    label_list = []

    global training_attributes
    global training_class_labels
    global testing_attributes
    global testing_class_labels

    reader=csv.reader(open(os.path.join(path_to_data, "Train/x_train.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in columns 0-561 of this attributes only file
            attribute_list.append(list(row[i] for i in (range(0,561))))

    reader=csv.reader(open(os.path.join(path_to_data, "Train/y_train.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # class label in column 1 of this labels only file
            label_list.append(row[0])

    training_attributes=np.array(attribute_list).astype(np.float32)
    training_class_labels=np.array(label_list).astype(np.float32)

    # Testing data - as currently split

    attribute_list = []
    label_list = []

    reader=csv.reader(open(os.path.join(path_to_data, "Test/x_test.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in columns 0-561 of this attributes only file
            attribute_list.append(list(row[i] for i in (range(0,561))))

    reader=csv.reader(open(os.path.join(path_to_data, "Test/y_test.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # class label in column 1 of this labels only file
            label_list.append(row[0])

    testing_attributes=np.array(attribute_list).astype(np.float32)
    testing_class_labels=np.array(label_list).astype(np.float32)

        ###########  test output for sanity

    print(training_attributes)
    print(len(training_attributes))
    print(training_class_labels)
    print(len(training_class_labels))

    print(testing_attributes)
    print(len(testing_attributes))
    print(testing_class_labels)
    print(len(testing_class_labels))

def read_data_nn():

    attribute_list = []
    label_list = []
    nn_outputs_list = []

    global training_attributes
    global training_class_labels
    global testing_attributes
    global testing_class_labels
    global training_nn_outputs
    global testing_nn_outputs ##Potentially not needed

    reader=csv.reader(open(os.path.join(path_to_data, "Train/x_train.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in columns 0-561 of this attributes only file
            attribute_list.append(list(row[i] for i in (range(0,561))))
            # attribute_list.append(list(row[i] for i in (list(range(0,57)))))


    reader=csv.reader(open(os.path.join(path_to_data, "Train/y_train.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # class label in column 1 of this labels only file
            label_list.append(row[0])
            nn_outputs_list.append(class_label_to_nn_output(row[0], len(classes), True, 1))


    training_attributes=np.array(attribute_list).astype(np.float32)
    training_class_labels=np.array(label_list).astype(np.float32)
    training_nn_outputs=np.array(nn_outputs_list).astype(np.float32)


    # Testing data - as currently split

    attribute_list = []
    label_list = []
    nn_outputs_list = []

    reader=csv.reader(open(os.path.join(path_to_data, "Test/x_test.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in columns 0-561 of this attributes only file
            attribute_list.append(list(row[i] for i in (range(0,561))))

    reader=csv.reader(open(os.path.join(path_to_data, "Test/y_test.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # class label in column 1 of this labels only file
            label_list.append(row[0])
            nn_outputs_list.append(class_label_to_nn_output(row[0], len(classes), True, 1)) ##Potentially not needed


    testing_attributes=np.array(attribute_list).astype(np.float32)
    testing_class_labels=np.array(label_list).astype(np.float32)
    testing_nn_outputs=np.array(nn_outputs_list).astype(np.float32) ##Potentially not needed


        ###########  test output for sanity

    print(training_attributes)
    print(len(training_attributes))
    print(training_class_labels)
    print(len(training_class_labels))

    print(testing_attributes)
    print(len(testing_attributes))
    print(testing_class_labels)
    print(len(testing_class_labels))




##this is the code for the original nn class
    if(False):
        ########### Load Training and Testing Data Sets

        # load training data set

        reader=csv.reader(open("spambase.train","rt", encoding='ascii'),delimiter=',')


        attribute_list = []
        label_list = []
        nn_outputs_list = []

        #### N.B there is a change in the loader here (compared to other examples)

        for row in reader:
                # attributes in columns 0-56, class label in last column,
                attribute_list.append(list(row[i] for i in (list(range(0,57)))))
                label_list.append(row[57])
                nn_outputs_list.append(class_label_to_nn_output(row[57], len(classes), True, 1))

        training_attributes=np.array(attribute_list).astype(np.float32)
        training_class_labels=np.array(label_list).astype(np.float32)
        training_nn_outputs=np.array(nn_outputs_list).astype(np.float32)

        # load testing data set

        reader=csv.reader(open("spambase.test","rt", encoding='ascii'),delimiter=',')

        attribute_list = []
        label_list = []
        nn_outputs_list = []

        for row in reader:
                # attributes in columns 0-56, class label in last column,
                attribute_list.append(list(row[i] for i in (list(range(0,57)))))
                label_list.append(row[57])

        testing_attributes=np.array(attribute_list).astype(np.float32)
        testing_class_labels=np.array(label_list).astype(np.float32)    

        ##################################################################### 



def dtree1():
    ############ Perform Training -- Decision Tree

    # define decision tree object

    dtree = cv2.ml.DTrees_create();

    # set parameters (changing may or may not change results)

    dtree.setCVFolds(1);       # the number of cross-validation folds/iterations - fix at 1
    dtree.setMaxCategories(25); # max number of categories (use sub-optimal algorithm for larger numbers)
    dtree.setMaxDepth(10);       # max tree depth
    dtree.setMinSampleCount(25); # min sample count
    dtree.setPriors(np.float32(np.ones(12)));  # the array of priors, the bigger weight, the more attention to the assoc. class
                                         #  (i.e. a case will be judjed to be maligant with bigger chance))
    dtree.setRegressionAccuracy(0);      # regression accuracy: N/A here
    dtree.setTruncatePrunedTree(True);   # throw away the pruned tree branches
    dtree.setUse1SERule(True);      # use 1SE rule => smaller tree
    dtree.setUseSurrogates(False);  # compute surrogate split, no missing data

    # specify that the types of our attributes is ordered with a categorical class output
    # and we have 7 of them (6 attributes + 1 class label)

    var_types = np.array([cv2.ml.VAR_NUMERICAL] * 561 + [cv2.ml.VAR_CATEGORICAL], np.uint8)

    # train decision tree object

    dtree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int), varType = var_types));

    ############ Perform Testing -- Decision Tree

    correct = 0
    wrong = 0

    # for each testing example

    for i in range(0, len(testing_attributes[:,0])) :

        # perform decision tree prediction (i.e. classification)

        _, result = dtree.predict(testing_attributes[i,:], cv2.ml.ROW_SAMPLE);

        # and for undocumented reasons take the first element of the resulting array
        # as the result

        print("Test data example : {} : result =  {}".format((i+1), inv_classes[int(result[0])]));

        # record results as tp/tn/fp/fn

        if (result[0] == testing_class_labels[i]) : correct+=1
        elif (result[0] != testing_class_labels[i]) : wrong+=1

    # output summmary statistics

    total = correct + wrong

    print();
    print("Testing Data Set Performance Summary");
    print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));

    #####################################################################

def dtree2(): 
    ############ Perform Training -- Decision Tree

    # define decision tree object

    dtree = cv2.ml.DTrees_create();

    # set parameters (changing may or may not change results)

    dtree.setCVFolds(1);       # the number of cross-validation folds/iterations - fixed at 1
    dtree.setMaxCategories(40); # max number of categories (use sub-optimal algorithm for larger numbers) #15 default
    dtree.setMaxDepth(13);       # max tree depth #8 default
    dtree.setMinSampleCount(96); # min sample count
    dtree.setPriors(np.float32(np.ones(12)));  # the array of priors, the bigger weight, the more attention to the assoc. class
                                         #  (i.e. a case will be judjed to be maligant with bigger chance))
    dtree.setRegressionAccuracy(0);      # regression accuracy: N/A here
    dtree.setTruncatePrunedTree(True);   # throw away the pruned tree branches
    dtree.setUse1SERule(True);      # use 1SE rule => smaller tree
    dtree.setUseSurrogates(False);  # compute surrogate split, no missing data

    # specify that the types of our attributes is numerical with a categorical output
    # and we have 31 of them (30 attributes + 1 class label)

    var_types = np.array([cv2.ml.VAR_NUMERICAL] * 561 + [cv2.ml.VAR_CATEGORICAL], np.uint8)

    # train decision tree object

    print("ready");

    dtree.train(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int), varType = var_types));

    ############ Perform Testing -- Decision Tree


    class_prediction_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_correct_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_incorrect_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    # for each testing example

    # for label_string, value in class_prediction_count.items():

    predicted_class_labels = np.empty_like(testing_attributes[:,0])

    #testing_attributes is 3162 rows by 561 columns
    for i in range(0, len(testing_attributes[:,0])):

        # perform decision tree prediction (i.e. classification)

        _, result = dtree.predict(testing_attributes[i,:]) #based on all of the attributes in each column, predict the result

        predicted_class_labels[i] = result[0]


        print("Test data example : {} : result =  {}".format((i+1), inv_classes[int(result[0])]));

        ##Code here: gets result[0] gives '1', '2' '3' etc depending on classifier.
        ##inv_classes[X] gets the 'name' of the matching '1' '2' '3' if it exists
        ##class_prediction_count[X] gets the value of the class count and adds one 
        class_prediction_count[inv_classes[int(result[0])]] += 1 
        # print(class_prediction_count) 
        if(result[0] == testing_class_labels[i]):
            class_correct_count[inv_classes[int(result[0])]] += 1 
        else:
            class_incorrect_count[inv_classes[int(result[0])]] += 1 
        ##Sanity check - add up all counts, should = length of data
    number_of_results = 0
    number_of_correct = 0
    for _, value in class_correct_count.items():
        number_of_results += value
        number_of_correct += value
    for _, value in class_incorrect_count.items():
        number_of_results += value

    print("Number of results = ", number_of_results, "\nNumber of data rows = ", len(testing_attributes[:,0]))
    print("class_prediction_count: ", class_prediction_count)
    print("class_correct_count: ", class_correct_count)
    print("class_incorrect_count: ", class_incorrect_count)

    # print("Accuracy: ", number_of_correct/number_of_results)
    print("Accuracy : {}%".format(str(round(number_of_correct/number_of_results * 100, 2))))

    return (predicted_class_labels, testing_class_labels)
        
        # record results as tp/tn/fp/fn

    #     if (result[0] == testing_class_labels[i] == 1) : tp+=1
    #     elif (result[0] == testing_class_labels[i] == 0) : tn+=1
    #     elif (result[0] != testing_class_labels[i]) :
    #         if ((result[0] == 1) and (testing_class_labels[i] == 0)) : fp+=1
    #         elif ((result[0] == 0) and (testing_class_labels[i] == 1)) : fn+=1

    # # output summmary statistics

    # total = tp + tn + fp + fn
    # correct = tp + tn
    # wrong = fp + fn

    # print();
    # print("Testing Data Set Performance Summary");
    # print("TP : {}%".format(str(round((tp / float(total)) * 100, 2))));
    # print("TN : {}%".format(str(round((tn / float(total)) * 100, 2))));
    # print("FP : {}%".format(str(round((fp / float(total)) * 100, 2))));
    # print("FN : {}%".format(str(round((fn / float(total)) * 100, 2))));
    # print("Total Correct : {}%".format(round((correct / float(total)) * 100, 2)));
    # print("Total Wrong : {}%".format(round((wrong / float(total)) * 100, 2)));

    #####################################################################

def knn():
    ############ Perform Training -- k-NN

    # define kNN object

    knn = cv2.ml.KNearest_create();

    # set to use BRUTE_FORCE neighbour search as KNEAREST_KDTREE seems to  break
    # on this data set (may not for others - http://code.opencv.org/issues/2661)

    knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE);

    # set default 3, can be changed at query time in predict() call

    knn.setDefaultK(3);

    # set up classification, turning off regression

    knn.setIsClassifier(True);

    # perform training of k-NN

    knn.train(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels);

    ############ Perform Testing -- k-NN


    # for each testing example
    class_prediction_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_correct_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_incorrect_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}


    for i in range(0, len(testing_attributes[:,0])) :

        # perform k-NN prediction (i.e. classification)

        # (to get around some kind of OpenCV python interface bug, vertically stack the
        #  example with a second row of zeros of the same size and type which is ignored).

        sample = np.vstack((testing_attributes[i,:],
                            np.zeros(len(testing_attributes[i,:])).astype(np.float32)))

        # now do the prediction returning the result, results (ignored) and also the responses
        # + distances of each of the k nearest neighbours
        # N.B. k at classification time must be < maxK from earlier training

        _, result, neigh_respones, distances = knn.findNearest(sample, k = 3);

        # print "Test data example : " + str(i + 1) + " : result = " + str(classes[int(result[0])])
        # print("Test data example : {} : result =  {}".format((i+1), classes[(int(results[0]))]));


        # _, result = svm.predict(sample, cv2.ml.ROW_SAMPLE);

            # print("Test data example : {:20s} : result =  {:20s} : actual = {:20s} ".format( (i+1), 
            #                                                                      inv_classes[int(result[0])], 
            #                                                                      inv_classes[testing_class_labels[i]]))


            ##Code here: gets result[0] gives '1', '2' '3' etc depending on classifier.
            ##inv_classes[X] gets the 'name' of the matching '1' '2' '3' if it exists
            ##class_prediction_count[X] gets the value of the class count and adds one 
        class_prediction_count[inv_classes[int(result[0])]] += 1 
        # print(class_prediction_count) 
        if(result[0] == testing_class_labels[i]):
            class_correct_count[inv_classes[int(result[0])]] += 1 
            print("Test data example : {:5} : result =  {:20s} : actual = {:20s} -----> CORRECT".format( (i+1), 
                                                                             inv_classes[int(result[0])], 
                                                                             inv_classes[testing_class_labels[i]]))

        else:
            class_incorrect_count[inv_classes[int(result[0])]] += 1 
            print("Test data example : {:5} : result =  {:20s} : actual = {:20s} -----> INCORRECT".format( (i+1), 
                                                                             inv_classes[int(result[0])], 
                                                                             inv_classes[testing_class_labels[i]]))

        ##Sanity check - add up all counts, should = length of data

    number_of_results = 0
    number_of_correct = 0
    for _, value in class_correct_count.items():
        number_of_results += value
        number_of_correct += value
    for _, value in class_incorrect_count.items():
        number_of_results += value

    print("Number of results = ", number_of_results, "\nNumber of data rows = ", len(testing_attributes[:,0]))
    print("class_prediction_count: ", class_prediction_count)
    print("class_correct_count: ", class_correct_count)
    print("class_incorrect_count: ", class_incorrect_count)

    # print("Accuracy: ", number_of_correct/number_of_results)
    print("Accuracy : {}%".format(str(round(number_of_correct/number_of_results * 100, 2))))

def svm():  
    
    ############ Perform Training -- SVM

    print("got to svm training")

    # define SVM object

    use_svm_autotrain = False;
    ## use SVM auto-training (grid search)
    # if available in python bindings; see open issue: https://github.com/opencv/opencv/issues/7224

    svm = cv2.ml.SVM_create();

    # set kernel
    # choices : # SVM_LINEAR / SVM_RBF / SVM_POLY / SVM_SIGMOID / SVM_CHI2 / SVM_INTER

    svm.setKernel(cv2.ml.SVM_LINEAR);

    # set parameters (some specific to certain kernels)

    svm.setC(10); # penalty constant on margin optimization
    svm.setType(cv2.ml.SVM_C_SVC); # multiple class (2 or more) classification
    # svm.setType(cv2.ml.SVM_NU_SVC); # multiple class (2 or more) classification

    svm.setGamma(0.5); # used for SVM_RBF kernel only, otherwise has no effect
    svm.setDegree(3);  # used for SVM_POLY kernel only, otherwise has no effect
    # svm.setNu(0.5)


    # set the relative weights importance of each class for use with penalty term

    svm.setClassWeights(np.ones(12)); ##was 26 for alphabet

    # define and train svm object

    if (use_svm_autotrain) :

        # use automatic grid search across the parameter space of kernel specified above
        # (ignoring kernel parameters set previously)

        # if available in python bindings; see open issue: https://github.com/opencv/opencv/issues/7224

        svm.trainAuto(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int)), kFold=2);
    else :

        # use kernel specified above with kernel parameters set previously

        svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels.astype(int));

    ############ Perform Testing -- SVM

    class_prediction_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_correct_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_incorrect_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    # for each testing example

    # for label_string, value in class_prediction_count.items():

    predicted_class_labels = np.empty_like(testing_attributes[:,0])

    #testing_attributes is 3162 rows by 561 columns
    for i in range(0, len(testing_attributes[:,0])):

        ######################SVMM
        # for i in range(0, len(testing_attributes[:,0])) :

        # (to get around some kind of OpenCV python interface bug, vertically stack the
        #  example with a second row of zeros of the same size and type which is ignored).

        sample = np.vstack((testing_attributes[i,:], np.zeros(len(testing_attributes[i,:])).astype(np.float32)));

        # perform SVM prediction (i.e. classification)

        _, result = svm.predict(sample, cv2.ml.ROW_SAMPLE); #result is of the form [label_index, total labels]

        predicted_class_labels[i] = result[0] ##Add the result to the array of predicted labels

        # print("Test data example : {:20s} : result =  {:20s} : actual = {:20s} ".format( (i+1), 
        #                                                                      inv_classes[int(result[0])], 
        #                                                                      inv_classes[testing_class_labels[i]]))

        ##Code here: gets result[0] gives '1', '2' '3' etc depending on classifier.
        ##inv_classes[X] gets the 'name' of the matching '1' '2' '3' if it exists
        ##class_prediction_count[X] gets the value of the class count and adds one 
        class_prediction_count[inv_classes[int(result[0])]] += 1 
        # print(class_prediction_count) 
        if(result[0] == testing_class_labels[i]):
            class_correct_count[inv_classes[int(result[0])]] += 1 
            # print("Test data example : {:5} : result =  {:20s} : actual = {:20s} -----> CORRECT".format( (i+1), 
            #                                                                  inv_classes[int(result[0])], 
            #                                                                  inv_classes[testing_class_labels[i]]))

        else:
            class_incorrect_count[inv_classes[int(result[0])]] += 1 
            # print("Test data example : {:5} : result =  {:20s} : actual = {:20s} -----> INCORRECT".format( (i+1), 
            #                                                                  inv_classes[int(result[0])], 
            #                                                                  inv_classes[testing_class_labels[i]]))

    ##Sanity check - add up all counts, should = length of data

    number_of_results = 0
    number_of_correct = 0
    for _, value in class_correct_count.items():
        number_of_results += value
        number_of_correct += value
    for _, value in class_incorrect_count.items():
        number_of_results += value

    print("Number of results = ", number_of_results, "\nNumber of data rows = ", len(testing_attributes[:,0]))
    print("class_prediction_count: ", class_prediction_count)
    print("class_correct_count: ", class_correct_count)
    print("class_incorrect_count: ", class_incorrect_count)

    # print("Accuracy: ", number_of_correct/number_of_results)
    print("Accuracy : {}%".format(str(round(number_of_correct/number_of_results * 100, 2))))

    return (predicted_class_labels, testing_class_labels)

def neuralnetworks():
    ############ Perform Training -- Neural Network

    # create the network object

    nnetwork = cv2.ml.ANN_MLP_create();

    # define number of layers, sizes of layers and train neural network
    # neural networks only support numerical inputs (convert any categorical inputs)

    # set the network to be 2 layer 57->10->2
    # - one input node per attribute in a sample
    # - 10 hidden nodes
    # - one output node per class
    # defined by the column vector layer_sizes

    num_hidden_layers = 5
    layer_sizes = np.int32([561, num_hidden_layers, len(classes)]); # format = [inputs, hidden layer n ..., output]
    nnetwork.setLayerSizes(layer_sizes);

    # create the network using a sigmoid function with alpha and beta
    # parameters = 1 specified respectively (standard sigmoid)

    nnetwork.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1, 1);

    # available activation functions = (cv2.ml.ANN_MLP_SIGMOID_SYM or cv2.ml.ANN_MLP_IDENTITY, cv2.ml.ANN_MLP_GAUSSIAN)

    # specify stopping criteria and backpropogation for training

    nnetwork.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP);
    nnetwork.setBackpropMomentumScale(0.1);
    nnetwork.setBackpropWeightScale(0.1);

    nnetwork.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 1000, 0.001))

            ## N.B. The OpenCV neural network (MLP) implementation does not
            ## support categorical variable output explicitly unlike the
            ## other OpenCV ML classes.
            ## Instead, following the traditional approach of neural networks,
            ## the output class label is formed by we a binary vector that
            ## corresponds the desired output layer result for a given class
            ## e.g. {0, 0 ... 1, 0, 0} components (one element by class) where
            ## an entry "1" in the i-th vector position correspondes to a class
            ## label for class i
            ## for optimal performance with the OpenCV intepretation of the sigmoid
            ## we use {-1, -1 ... 1, -1, -1}

            ## prior to training we must construct these output layer responses
            ## from our conventional class labels (carried out by class_label_to_nn_output()

    # train the neural network (using training data)

    nnetwork.train(training_attributes, cv2.ml.ROW_SAMPLE, training_nn_outputs);

    ############ Perform Testing -- Neural Network
    # for each testing example

    predicted_class_labels = np.empty_like(testing_attributes[:,0])

    for i in range(0, len(testing_attributes[:,0])) :

        # perform neural network prediction (i.e. classification)

        # (to get around some kind of OpenCV python interface bug, vertically stack the
        #  example with a second row of zeros of the same size and type which is ignored).

        sample = np.vstack((testing_attributes[i,:],
                            np.zeros(len(testing_attributes[i,:])).astype(np.float32)));

        retrval,output_layer_responses = nnetwork.predict(sample);

        # the class label c (result) is the index of the most
        # +ve of the output layer responses (from the first of the two examples in the stack)

        result = np.argmax(output_layer_responses[0]) +1; ####Corrected with +1 for the -1 in class to nn output method

        predicted_class_labels[i] = result

    return (predicted_class_labels, testing_class_labels)



def print_accuracy_results(predicted_labels, true_labels):

    if(len(predicted_labels)!=len(true_labels)):
        print("WARNING: PREDICTED_LABELS MUST BE SAME LENGTH AS TRUE_LABELS")
        ##Throw exception

    ##initialise counting dictionaries
    class_prediction_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_correct_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    class_incorrect_count = {'WALKING' : 0, 'WALKING_UPSTAIRS' : 0, 'WALKING_DOWNSTAIRS' : 0, 'SITTING' : 0, 'STANDING' : 0, 'LAYING' : 0,'STAND_TO_SIT' : 0, 'SIT_TO_STAND' : 0, 'SIT_TO_LIE' : 0, 'LIE_TO_SIT' : 0, 'STAND_TO_LIE' : 0, 'LIE_TO_STAND' :0}
    

    for i in range(0, len(true_labels)):
        class_prediction_count[inv_classes[int(predicted_labels[i])]] += 1 
        ##Code here: gets result[0] gives '1', '2' '3' etc depending on classifier.
        ##inv_classes[X] gets the 'name' of the matching '1' '2' '3' if it exists
        ##class_prediction_count[X] gets the value of the class count and adds one
        if(predicted_labels[i] == true_labels[i]):
            class_correct_count[inv_classes[int(predicted_labels[i])]] += 1 
            if((i+1)%100 == 0):
                print("Test data example : {:5} : result =  {:20s} : actual = {:20s} -----> CORRECT".format( (i+1), 
                                                                             inv_classes[int(predicted_labels[i])], 
                                                                             inv_classes[true_labels[i]]))

        else:
            class_incorrect_count[inv_classes[int(predicted_labels[i])]] += 1 
            if((i+1)%100 == 0):
                print("Test data example : {:5} : result =  {:20s} : actual = {:20s} -----> INCORRECT".format( (i+1), 
                                                                             inv_classes[int(predicted_labels[i])], 
                                                                             inv_classes[true_labels[i]]))

    ##Sanity check - add up all counts, should = length of data

    number_of_results = 0
    number_of_correct = 0
    for _, value in class_correct_count.items():
        number_of_results += value
        number_of_correct += value
    for _, value in class_incorrect_count.items():
        number_of_results += value

    print("Number of results = ", number_of_results, "\nNumber of data rows = ", len(testing_attributes[:,0]))
    print("class_prediction_count: ", class_prediction_count)
    print("class_correct_count: ", class_correct_count)
    print("class_incorrect_count: ", class_incorrect_count)

    # print("Accuracy: ", number_of_correct/number_of_results)
    print("Accuracy : {}%".format(str(round(number_of_correct/number_of_results * 100, 2))))


def gen_report(predicted_labels, true_labels):
    # cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)
    cm = confusion_matrix(true_labels, predicted_labels)
    cr = classification_report(true_labels, predicted_labels)
    print(cm)
    print(cr)
    return (cm, cr)



if __name__ == "__main__":
    # read_data()

    # read_data_nn()
    # true_labels, predicted_labels = neuralnetworks()
    # print_accuracy_results(true_labels, predicted_labels)
    # cm, cr = gen_report(true_labels, predicted_labels)
    # skl_report.plot_classification_report(cr)
    # pyplot.show()

    read_data()
    true_labels, predicted_labels = dtree2()
    cm, cr = gen_report(true_labels, predicted_labels)
    skl_report.plot_classification_report(cr)
    pyplot.show()

    


