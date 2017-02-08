import csv
import cv2
import os
import numpy as np



def svm(classes, inv_classes, training_attributes, testing_attributes, training_class_labels, testing_class_labels):  
    
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

    # svm.setClassWeights(np.ones(12)); ##was 26 for alphabet
    svm.setClassWeights(np.ones(len(classes)))

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
