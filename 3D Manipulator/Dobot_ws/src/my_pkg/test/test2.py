import random
import numpy as np
import math

num_of_mat = 2000
D = 0.008
k = math.pi*400/3
N = 8

def generate_matrix(size_mat):
    matrix = []
    for i in range(size_mat*size_mat):
        index = random.random()
        index = 1 if index >= 0.5 else 0
        matrix.append(index)
    matrix = np.array(matrix)
    matrix = matrix.reshape((size_mat, size_mat))
    return matrix

def find_same(mat_list, mat):
    z = 0
    for i in range(len(mat_list)):
        if (mat == mat_list[i]).all():
            z += 0
        else:
            z += 1
    if z == len(mat_list):
        return True
    else:
        return False


def main():
    matrix_list = []
    matrix_value = []

    for i_mat in range(num_of_mat):
        print(i_mat)
        while 1:
            matrix = generate_matrix(N)
            same = find_same(matrix_list, matrix)
            if same:
                break
        print(matrix)

        angel_list = np.zeros((360, 91))
        for theta in range(360):
            theta_rad = theta*2*math.pi/360
            for phi in range(91):
                phi_rad = phi*2*math.pi/360
                
                sum_sin = 0
                sum_cos = 0
                for m in range(N):
                    for n in range(N):
                        x = matrix[m][n]*math.pi + \
                            k*D*math.sin(theta_rad)*((m-0.5)*math.cos(phi_rad)+(n-0.5)*math.sin(phi_rad))
                        sum_sin += math.sin(x)
                        sum_cos += math.cos(x)
                        # f_thets_phi = (math.cos(x)**2+math.sin(x)**2)**0.5
                        # sum_f += f_thets_phi
                sum_f = (math.cos(sum_cos)**2+math.sin(sum_sin)**2)**0.5
                angel_list[theta][phi] = sum_f
        
        print(np.amax(angel_list))
        print(np.amin(angel_list))
        matrix_list.append(matrix)
        matrix_value.append(np.amax(angel_list))

    best_index = matrix_value.index(min(matrix_value))
    best_matrix = matrix_list[best_index]
    print(best_matrix)


if __name__ == '__main__':
    main()