import numpy as np
import math

test_arr = np.arange(0,10929)
test_arr = np.arange(0,109)

print(len(test_arr))
print(test_arr)

k = 5 ## 100 must be divisible by k

for i in range(k):
    # i = 0,1,2,3,4,5
    ##check 100 is divisible by k
    if(100%k!=0): 
        # print("Error: 100 must be divisible by k")
        raise ValueError("Error: 100 must be divisible by k")

    N = math.floor(100/k)
    print(N)

    first_percentile = (i*N)/100
    second_percentile = (i+1)*N/100
    len_arr = len(test_arr)
    first_index = int(len_arr*first_percentile)
    second_index = int(len_arr*second_percentile)

    testing_array = test_arr[first_index:second_index]
    training_array= np.concatenate(test_arr[0:first_index], test_arr[second_index:len_arr]) ##Everything but the above 



    print(i*N, (i+1)*N)
    print(new_array)
    print("len_new_arr", len(new_array))

