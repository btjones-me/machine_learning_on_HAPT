import csv
import cv2
import os
import numpy as np

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

def class_label_to_nn_output(label, max_classes, is_sigmoid, value):
    ########### construct output layer

    # expand training responses defined as class labels {0,1...,N} to output layer
    # responses for the OpenCV MLP (Neural Network) implementation such that class
    # label c becomes {0,0,0, ... 1, ...0} where the c-th entry is the only non-zero
    # entry (equal to "value", conventionally = 1) in the N-length vector

    # labels : a row vector of class label transformed to {0,0,0, ... 1, ...0}
    # max_classes : maximum class label
    # value: value use to label the class response in the output layer vector
    # sigmoid : {true | false} - return {-value,....value,....-value} instead for
    #           optimal use with OpenCV sigmoid function

    if (is_sigmoid):
        output = np.ones(max_classes).astype(np.float32) * (-1 * value)
        output[int(label)-1] = value
    else:
        output = np.zeros(max_classes).astype(np.float32)
        output[int(label)-1] = value

    return output
