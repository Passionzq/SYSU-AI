def read_file(filename : str):
    init_matrix = []
    with open(filename) as f:
        for ln in f:
            init_matrix.append(ln.split())
    
    output = []
    for i in init_matrix:
        row = []
        for j in i:
            row.append(ord(j) - ord('0'))
        output.append(row)

    return output

def write_file(filename : str, matrix):
    with open(filename, "a") as f:
        for ln in matrix:
            for i in range(0,9):
                f.write(str(ln[i]) + " ")
                if(8 == i):
                    f.write("\n")
        # f.write("-----------------\n")
            
def print_matrix(matrix):
    for ln in matrix:
        for x in ln:
            print(str(x) + " ",end="")
        print()
    print("-----------------")
