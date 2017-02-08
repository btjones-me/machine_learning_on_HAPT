#####################################################################

# Code to load, cross validate and run machine learning examples 
# For use with test / training datasets : HAPT-data-set-DU

# Author : Benjamin Jones, b.t.jones@durham.ac.uk, zkbb46
# Based on code from the CV2 open source library and code provided by Dr Toby Breckon, 
# https://github.com/tobybreckon/python-examples-ml

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import csv
import cv2
import os
import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import plot_sklearn_report as skl_report
import matplotlib.pyplot as pyplot

import svm_code
import stats_code
import plot_report

########### Define classes

classes = {'WALKING' : 1, 'WALKING_UPSTAIRS' : 2, 'WALKING_DOWNSTAIRS' : 3, 'SITTING' : 4, 'STANDING' : 5, 'LAYING' : 6,'STAND_TO_SIT' : 7, 'SIT_TO_STAND' : 8, 'SIT_TO_LIE' : 9, 'LIE_TO_SIT' : 10, 'STAND_TO_LIE' : 11, 'LIE_TO_STAND' :12}
inv_classes = {v: k for k, v in classes.items()}

print(classes)
path_to_data = "..//HAPT-data-set-DU" # edit this

##Read in the normal dataset
##Combine the normal dataset and shuffle it with the seed being equal to 500
##Read in the strat k fold indexes generated from sklearn in the past
##Generate the strat k fold datasets and run against different algorithms

def read_combine_randomize_data():
    ##Reads in data from HAPT data set, combines and randomises the output.

    # training_attributes_local = []
    # training_class_labels_local = []
    # testing_attributes_local = []
    # testing_class_labels_local = []

    print("Reading in original HAPT data set")

    total_attributes = []
    total_labels = []

    attribute_list = []
    label_list = []

    reader=csv.reader(open(os.path.join(path_to_data, "Train/x_train.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # attributes in columns 0-561 of this attributes only file
            attribute_list.append(list(row[i] for i in (range(0,561))))

    reader=csv.reader(open(os.path.join(path_to_data, "Train/y_train.txt"),"rt", encoding='ascii'),delimiter=' ')
    for row in reader:
            # class label in column 1 of this labels only file
            label_list.append(row[0])

    total_attributes = np.array(attribute_list).astype(np.float32)
    total_labels = np.array(label_list).astype(np.float32)

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

    # testing_attributes_local = np.array(attribute_list).astype(np.float32)
    # testing_class_labels_local = np.array(label_list).astype(np.float32)

    total_attributes = np.concatenate((total_attributes, np.array(attribute_list).astype(np.float32)), axis=0)
    total_labels = np.concatenate((total_labels, np.array(label_list).astype(np.float32)), axis=0)

    # print(total_attributes)
    # print(len(total_attributes))
    # print(total_labels)
    # print(len(total_labels))

    ##Randomise the data on return by shuffling the same way.

    print("Returning concatenated and shuffled HAPT data set")
    ##Do not modify the seed. This must remain constant for dataset to be consistent.
    return (randomize(total_attributes, seed=500), randomize(total_labels, seed=500))

def read_indexes(num_k_folds = 10):

    print("Reading in indexes")

    train_set = [] ###This will be an array of length equal to number of k folds 
    test_set = [] ###This will be an array of length equal to number of k folds 

    for i in range(num_k_folds) : 
       
        reader_train=csv.reader(open("./test_data/train_indexes_" + str(i) + ".data","r", encoding='ascii'),delimiter=' ')
        entry_list = []
        for row in reader_train:
            entry_list.append(row[0])
            # print(row)
        train_set.append(np.array(entry_list).astype(int))

        reader_test=csv.reader(open("./test_data/test_indexes_" + str(i) + ".data","r", encoding='ascii'),delimiter=' ')
        entry_list = []
        for row in reader_test:
            entry_list.append(row[0])
        test_set.append(np.array(entry_list).astype(int))

    return (train_set, test_set)

def randomize(a_list, seed):
    np.random.seed(seed) 
    np.random.shuffle(a_list)
    return a_list

def check_length(total_attributes, total_labels, train_set, test_set):
    ##If all has worked as planned, the combined training set in HAPT data should match length of indexes in k fold CV.
    for i in range(len(train_set)):    
        if(len(total_attributes) == len(train_set[i]) + len(test_set[i])):
            print("CONFIRMED: len(total_attributes) == len(train_set["+str(i)+"]) + len(test_set["+str(i)+"])")
        else:
            errmsg = "ERROR: lengths unequal- len(total_attributes) != len(train_set["+str(i)+"]) + len(test_set["+str(i)+"])"
            raise ValueError(errmsg)

def program_manager():
    total_attributes, total_labels = read_combine_randomize_data()
    train_set, test_set = read_indexes()

    for i in range(len(train_set)):
        training_attributes = total_attributes[train_set[i]]
        testing_attributes = total_attributes[test_set[i]]

        training_labels = total_labels[train_set[i]] 
        testing_labels = total_labels[test_set[i]]

# def svm(classes, inv_classes, training_attributes, testing_attributes, training_class_labels, testing_class_labels, *params=None):  

        predicted, testing = svm_code.svm(classes, inv_classes, training_attributes, testing_attributes, training_labels, testing_labels)

        stats_code.print_accuracy_results(predicted, testing) 
        # cm, cr = stats_code.gen_report(testing, predicted) ##Requires sklearn
        # plot_report.plot_classification_report(cr)
    # pyplot.show()




program_manager()


