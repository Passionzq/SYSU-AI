import r_w
import sys
import time

size = 9
judgement = False
file_id = 0
node_num = 0

def is_finished(matrix):
    for ln in matrix:
        s = set(ln)
        if len(s) != 9 or 0 in s:
            return False
    
    for i in range(0,9):
        temp = []
        for ln in matrix:
            temp.append(ln[i])
        s = set(temp)
        if len(s) != 9 or 0 in s:
            return False
    
    for i in range(0,3):
        for j in range(0,3):
            temp = []
            for x in range(0,3):
                for y in range(0,3):
                    temp.append(matrix[3*i+x][3*j+y])
            s = set(temp)
            if len(s) != 9 or 0 in s:
                return False
    return True

def is_valid(judge_matrix, x : int, y: int):
    if (judge_matrix[x][y] != 0):
        empty = []
        return empty

    judge = [0,0,0,0,0,0,0,0,0,0]
    
    for i in judge_matrix[x]:
        judge[i] = 1

    for i in range(0,9):
        judge[judge_matrix[i][y]] = 1
    
    for i in range(0,3):
        for j in range(0,3):
            judge[judge_matrix[x//3 * 3+i][y//3 * 3+j]] = 1
    
    count = 0
    valid_num = []
    for num in judge:
        if num == 0 and count != 0:
            valid_num.append(count)
        count = count + 1
    
    return valid_num

def rank(matrix):
    output = []
    for i in range(0,9):
        for j in range(0,9):
            if (0 == matrix[i][j]):
                temp = is_valid(matrix,i,j)
                temp.append(i*size+j)
                output.append(temp)
    output.sort(key = lambda i:len(i))
    return output

def brute_force(num, curr_matrix):
    global judgement
    if (judgement == True):
        return

    if (81 == num):
        if is_finished(curr_matrix):
            judgement = True 
            # print("The answer is: ")
            # r_w.print_matrix(curr_matrix)
            global file_id
            ans_name = "./data/BF/BFans" + str(file_id)+".txt"
            r_w.write_file(ans_name,curr_matrix)
        return

    global node_num
    node_num = node_num + 1
    x = num // size
    y = num % size

    if (0 == curr_matrix[x][y]):
        for k in range(0,9):
            curr_matrix[x][y] = k
            brute_force(num + 1, curr_matrix)
            curr_matrix[x][y] = 0
        return
    else:
        brute_force(num + 1, curr_matrix)

def back_tracking(num, curr_matrix):
    
    global judgement
    if (judgement == True):
        return
    
    if (81 == num):
        if is_finished(curr_matrix):
            judgement = True 
            # print("The answer is: ")
            # r_w.print_matrix(curr_matrix)
            global file_id
            ans_name = "./data/BT/BTans" + str(file_id)+".txt"
            r_w.write_file(ans_name,curr_matrix)
        return

    global node_num
    node_num = node_num + 1
    x = num // size
    y = num % size

    
    if (0 == curr_matrix[x][y]):
        for k in is_valid(curr_matrix,x,y):
            curr_matrix[x][y] = k
            back_tracking(num + 1, curr_matrix)
            curr_matrix[x][y] = 0
        return
    else:
        back_tracking(num + 1, curr_matrix)

def MRV(curr_matrix):
    global judgement
    if (judgement == True):
        return

    rank_matrix = rank(curr_matrix)
    if (0 == len(rank_matrix)):
        if is_finished(curr_matrix):
            judgement = True 
            # print("The answer is: ")
            # r_w.print_matrix(curr_matrix)
            global file_id
            ans_name = "./data/MRV/MRVans" + str(file_id)+".txt"
            r_w.write_file(ans_name,curr_matrix)
        return

    global node_num
    node_num = node_num + 1
    elem = rank_matrix[0]
    x = elem[-1] // size
    y = elem[-1] % size
    # print(elem,"\n(",x,",",y,")")

    for i in range(0,len(elem)-1):
        curr_matrix[x][y] = elem[i]
        MRV(curr_matrix)
        curr_matrix[x][y] = 0
    return


# sys.setrecursionlimit(900000000)
m = r_w.read_file("./data/example1.txt")
choice = input("1.BF\t2.BT\t3.MRV\t4.Quit\nPlease enter the method u want to use: ")
choice = int(choice)
if (choice not in range(1,5)):
    print("Invalid input")
else:
    for file_id in range(1,7):
        node_num = 0
        judgement = False
        example_name = "./data/example" + str(file_id) + ".txt"
        m = r_w.read_file(example_name)
        
        system_running_time_start = time.perf_counter()
        process_time_start = time.process_time()

        if (choice == 1):
            brute_force(0,m)
        elif (choice == 2):
            back_tracking(0,m)
        elif (choice == 3):
            MRV(m)
        else:
            print("Bye~")

        if (choice != 4):
            process_time_end = time.process_time()
            system_running_time_end = time.perf_counter()
            print("Example ",file_id)
            print("PROCESS time cost: ",process_time_end - process_time_start, "s\nSYSTEM running time cost: ",system_running_time_end - system_running_time_start,"s\nSearch Node num: ",node_num)
            print("--------------------------------")
                