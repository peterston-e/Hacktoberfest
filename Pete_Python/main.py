import numpy as np

def swap_rows(M, row_index_1, row_index_2): 
    M = M.copy()
   
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

def get_index_first_non_zero_value_from_column(M, column, starting_row):
    # Get the column array starting from the specified row
    column_array = M[starting_row:,column]
    for i, val in enumerate(column_array):

        if not np.isclose(val, 0, atol = 1e-5):
            
            index = i + starting_row
            return index
  
    return -1

def get_index_first_non_zero_value_from_row(M, row, augmented = False):

    # Create a copy to avoid modifying the original matrix
    M = M.copy()


    # If it is an augmented matrix, then ignore the constant values
    if augmented == True:
        # Isolating the coefficient matrix (removing the constant terms)
        M = M[:,:-1]
        
    # Get the desired row
    row_array = M[row]
    for i, val in enumerate(row_array):
        # If finds a non zero value, returns the index. Otherwise returns -1.
        if not np.isclose(val, 0, atol = 1e-5):
            return i
    return -1

def augmented_matrix(A, B):
   
    augmented_M = np.hstack((A,B))
    return augmented_M

def row_echelon_form(A, B):

    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0) == True:
        return 'Singular system'

    A = A.copy()
    B = B.copy()
   
    A = A.astype('float64')
    B = B.astype('float64')


    num_rows = len(A) 
    M = augmented_matrix(A, B)
    

    for row in range(num_rows):
        pivot_candidate = M[row, row]
        if np.isclose(pivot_candidate, 0) == True: 
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(M, row, row) 
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate) 
            pivot = M[row,row] 
        else:
            pivot = pivot_candidate 
     
        M[row] = 1/pivot * M[row]
       
        for j in range(row + 1, num_rows): 

            value_below_pivot = M[j,row]
            M[j] = M[j] - value_below_pivot*M[row]

    return M
            