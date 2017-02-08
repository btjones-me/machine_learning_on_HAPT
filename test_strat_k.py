import numpy as np
import csv
import cv2
import os
import matplotlib.pyplot as pyplot

path_to_data = "..//HAPT-data-set-DU" # edit this

training_attributes = 0
training_class_labels = 0
testing_attributes = 0
testing_class_labels = 0


test_a = 0 ##Used to confirm that the written output and read in input are the same
test_b = 0

def write_stratKfold(attributes, labels, n_splits):
    from sklearn.model_selection import StratifiedKFold ##Dependancy only needed in this method

    print("Beginning stratified k-fold cross validation")

    skf = StratifiedKFold(n_splits) ##Uses Sklearn to maintain original weighted label proportions when k-fold CVing data into n_splits.

    train_set = []
    test_set = []

    for i, data in enumerate(skf.split(attributes, labels)): ##sklearn does the splitting here
        train, test = data
        # print("%s %s" % (train, test))
        # print("********")
        train_set.append(train)
        test_set.append(test)

        # print("%s %s %s" % (len(train), len(test), len(train)+len(test)))

    # print(len(train_set))
    # print(len(train_set[0]))
    # print(train_set[0])
    global test_a
    test_a = np.concatenate((train_set[0], test_set[0]), axis=0) 

    print("Writing stratified k-fold validation indexes to " + str(n_splits) + " file(s)")

    for idx, train in enumerate(train_set):
        writer = csv.writer(open("./test_data/train_indexes_" + str(idx) + ".data", "w", encoding='ascii'), delimiter='\n', lineterminator='\n')
        writer.writerow(train)

    for idx, test in enumerate(test_set):
        writer = csv.writer(open("./test_data/test_indexes_" + str(idx) + ".data", "w", encoding='ascii'), delimiter='\n', lineterminator='\n')
        writer.writerow(test)

def read_indexes(num_files_of_each=10):

    print("Reading in indexes")

    train_set = []
    test_set = []

    for i in range(num_files_of_each): 
       
        reader_train=csv.reader(open("./test_data/train_indexes_" + str(i) + ".data","r", encoding='ascii'),delimiter=' ')
        entry_list = []
        for row in reader_train:
            entry_list.append(row[0])
            # print(row)
        train_set.append(np.array(entry_list).astype(np.float32))

        reader_test=csv.reader(open("./test_data/test_indexes_" + str(i) + ".data","r", encoding='ascii'),delimiter=' ')
        entry_list = []
        for row in reader_test:
            entry_list.append(row[0])
        test_set.append(np.array(entry_list).astype(np.float32))

    global test_a
    global test_b
    test_b = np.concatenate((train_set[0], test_set[0]), axis=0) 


    # print("\ntest_a_len: ", len(test_a))    
    # print("\ntest_b_len: ", len(test_b))
    # print("\ntest_a: ", test_a)    
    # print("\ntest_b: ", test_b)


    if(np.array_equal(test_a,test_b)):
        print("Confirmed: Array read in is equivalent to array written out")
    else:
        print("FACEPALM- Array read in was different to array written out.")

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

    print(total_attributes)
    print(len(total_attributes))
    print(total_labels)
    print(len(total_labels))

    ##Randomise the data on return by shuffling the same way.

    print("Returning concatenated and shuffled HAPT data set")
    ##Do not modify the seed. This must remain constant for dataset to be consistent.
    return (randomize(total_attributes, seed=500), randomize(total_labels, seed=500))


##In a for loop, read in different chunks of the data to the algorithm

# def k_fold_validation(total_attributes, total_labels, seed = 500):
#     ##You could stratify it by counting occurances in the label list and selecting n from each label
#     ##where n is the proportion of (labels/total)*(1/k) or something
#     ##seed is arbitrary, chosen so as not to coincide with other students using seeded k-fold

#     total_attributes = randomize(total_attributes, seed) #These should be mixed IN THE SAME WAY to ensure data is random
#     total_labels = randomize(total_labels, seed)


def randomize(a_list, seed):
    np.random.seed(seed) 
    np.random.shuffle(a_list)
    return a_list


# def read_data():
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
total_attributes, total_labels = read_combine_randomize_data()
write_stratKfold(total_attributes, total_labels, 10)
read_indexes()


