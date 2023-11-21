#!/usr/bin/env python3
import random
import numpy as np
# The maximum T_d is set to be 15 min, since every 15 min there will be an uplink
# message for locaiton update so the RRC connection will be automatically
# triggered and thus the new parameters setting can be delievered to the UE.
def genTraffic(record,cur_time):
    print(record,cur_time)
    if not record:
        state = 3600000
    else:
        if cur_time < record[0] and cur_time+3600000 >= record[0]:
            state = record[0]-cur_time
            record.pop(0)
        else:
            state = 3600000
    print(state)
    return state  
# Generate 1000*24 matrix for the Rth hourly scooter usage record 
def genrecord(R,output_numbers):
    rows = 100
    columns = 24
    probabilities = [int(x*rows)/rows for x in output_numbers[R]]
    gold = 0.3*max(probabilities)
    #gold  = 0.1
    #gold = np.percentile(probabilities, 10) 
    matrix = np.zeros((rows, columns))
  
    for col, prob in enumerate(probabilities): 
        while np.sum(matrix[:, col] != 0, axis=0) / matrix[:,col].shape[0] != prob:
            matrix[:, col] = np.random.choice([0, 1], size=(rows,), p=[1-prob, prob])     
    matrix[matrix == 1] = [random.randint(0, 60) for _ in range(int(np.sum(matrix)))]
    return matrix, gold

# Genreate traffic probability array store all the arrival
# of the downlink traffic in senconds.
def genrow(record): 
    res = []
    cnt = 0
    for i in range(1,len(record)):
        if record[i] != 0:
            res.append((cnt+record[i])*60000)
            cnt += 60
        else:
            cnt += 60
    return res
 
