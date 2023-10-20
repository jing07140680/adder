import numpy as np
import random
output_numbers = []
np.set_printoptions(threshold=np.inf)

with open('../scooterdata/actual.txt', 'r') as f:
    for line in f:
        numbers = line.strip().split()  # Assuming numbers are space-separated
        # Extract every 15th number starting from the first
        unique_numbers = numbers[::60]
        output_numbers.append([float(x) for x in unique_numbers])
        break
f.close()    
probabilities = output_numbers[0]
print(probabilities)
rows = 100
columns = 24
matrix = np.zeros((rows, columns))

for col, prob in enumerate(probabilities):
        matrix[:, col] = np.random.choice([0, 1], size=(rows,), p=[1-prob, prob])

# Replace 1s with random value from 0 to 60
matrix[matrix == 1] = [random.randint(0, 60) for _ in range(int(np.sum(matrix)))]

print(matrix)
