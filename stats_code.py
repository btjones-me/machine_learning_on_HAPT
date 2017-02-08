#####################################################################

# Code to produce statistical outputs

# For use with test / training datasets : HAPT-data-set-DU

# Author : Benjamin Jones, b.t.jones@durham.ac.uk, zkbb46
# Based on code from the CV2 open source library and code provided by Dr Toby Breckon, 
# https://github.com/tobybreckon/python-examples-ml

# Copyright (c) 2014 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

#!! WARNING !! - This code depends on the sklearn package. Will not run without it. 
#       This code is not necessary for the assignment as all results are outputted to text files.
#       The code is included for interest only. 

#####################################################################

import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as pyplot

classes = {'WALKING' : 1, 'WALKING_UPSTAIRS' : 2, 'WALKING_DOWNSTAIRS' : 3, 'SITTING' : 4, 'STANDING' : 5, 'LAYING' : 6,'STAND_TO_SIT' : 7, 'SIT_TO_STAND' : 8, 'SIT_TO_LIE' : 9, 'LIE_TO_SIT' : 10, 'STAND_TO_LIE' : 11, 'LIE_TO_STAND' :12}
inv_classes = {v: k for k, v in classes.items()}

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

    print("Number of results = ", number_of_results, "\nNumber of data rows = ", len(predicted_labels))
    print("class_prediction_count: ", class_prediction_count)
    print("class_correct_count: ", class_correct_count)
    print("class_incorrect_count: ", class_incorrect_count)

    # print("Accuracy: ", number_of_correct/number_of_results)
    print("Accuracy : {}%".format(str(round(number_of_correct/number_of_results * 100, 2))))


def gen_report(predicted_labels, true_labels):
    from sklearn.metrics import classification_report, confusion_matrix

    # cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)
    cm = confusion_matrix(true_labels, predicted_labels)
    cr = classification_report(true_labels, predicted_labels)
    print(cm)
    print(cr)
    return (cm, cr)