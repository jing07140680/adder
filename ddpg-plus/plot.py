import pickle
import numpy as np

def readin(pickle_file):
    # Open the file in binar y read mode
    with open(pickle_file, 'rb') as file:
        # Load the list from the file
        my_list = pickle.load(file)

        return my_list


L_10 = readin("L_10.pkl")
L_40 = readin("L_40.pkl")
E_10 = readin("E_10.pkl")
E_40 = readin("E_40.pkl")
L_edrx = readin("L_edrx.pkl")
E_edrx = readin("E_edrx.pkl")
L_AC = readin("L_AC.pkl")
E_AC = readin("E_AC.pkl")

sort_and_reshape = lambda x: np.sort((np.array(x)).reshape(-1))
sorted_data1 = sort_and_reshape(L_10)
sorted_data2 = sort_and_reshape(L_40)
E_10 = sort_and_reshape(E_10)
E_40 = sort_and_reshape(E_40)
sorted_data3 = sort_and_reshape(L_edrx)
sorted_data4 = sort_and_reshape(L_AC)
E_edrx = sort_and_reshape(E_edrx)

print(np.mean(E_edrx),np.mean(E_AC), np.mean(E_10), np.mean(E_40))
print(max(sorted_data3),max(E_AC), max(sorted_data1), max(sorted_data2))
