import pandas as pd

df = pd.read_csv("matrix_data_inv.csv")

def recursiveCleaner(matrix, count):
    if count <= 20:
        for i in range(len(matrix)):
            if matrix[i] not in ["0", "1", "2","3", "4", "5", "6", "7", "8", "9", "-"]:
                print(matrix[:i+1] + matrix[i:])
                matrix = matrix[:i] + matrix[i:]
                return recursiveCleaner(matrix, count+1)

    else:
        print(matrix)
        return matrix

for index, row in df.iterrows():
    recursiveCleaner(row["matrix"], 0)

    
print(df)

