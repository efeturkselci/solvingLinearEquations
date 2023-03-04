from __future__ import annotations
import numpy as np
from time import time

start = time() 

#Reading the matrices from a file
def readMatricesFromFile(filepath):    
    f = open(filepath, 'r')
    n = f.readline()
    t = n.split()        
    n = int(t[2])  
    f.readline()
            
    matrix = []
            
    t = f.readline().split()
    for i in range(n):
        rowsIteration = []
        for j in range(n):
            row, column, value = int(t[0]),int(t[1]), float(t[2]) #interpreting the indexes as rows, columns and values

            if(row == i and column == j):
                rowsIteration.append(value)
                t = f.readline().split()
            else:
                rowsIteration.append(0)
        matrix.append(rowsIteration)
    return matrix

#reading the solutions file in order to compare the final results
def readSolutions(filepath):    
    f = open(filepath, 'r')
    n = f.readline()
    t = n.split()        
    n = float(t[0])
    line = sum(1 for line in open(filepath)) #takes the count of lines in the file
           
    matrix = []
            
    for i in range(line):
        value = float(t[0])
        matrix.append(value)
        t = f.readline().split()
    return matrix            


class Matrix():
    #constructor
    def __init__(self, matrix, row, column):
        self.matrix = matrix
        self.row = row  
        self.column = column
        
    #display the matrix on the console with the elements, written for control purposes
    def displayMatrix(self):
        print(self.matrix)
    
    #addition of two matrices that have same dimensions, operator overloading
    def __add__(self, other):
        if(self.row == other.row and self.column == other.column):
            sol = []
            for i in range(self.row):
                rowsIteration = [] #iteration in first matrix's rows
                for j in range(self.column):
                    rowsIteration.append(self.matrix[i][j] + other.matrix[i][j]) 
                sol.append(rowsIteration)   
            return Matrix(sol, self.row, self.column)
        else:
            print("Matrix dimensions are different, can't operate addition")
                       

    #subtract operation between two matrices, that have same dimensions, operator overloading
    def __sub__(self, other):
        if (self.row == other.row and self.column == other.column):
            sol = []
            for i in range(self.row):
                rowsIteration = [] #iteration in first matrix's rows
                for j in range(self.column):
                    rowsIteration.append(self.matrix[i][j] - other.matrix[i][j])
                sol.append(rowsIteration)
            return Matrix(sol, self.row, self.column)
        else:
            print("Matrix dimensions are different, can't operate subtraction")
           
            
    #multiplication operation between two same dimensional matrices, operator overloading 
    def __mul__(self, other):
        if(self.column == other.row):
            sol = []
            for i in range(self.row):
                rowsIteration = []
                for j in range(self.column):
                    rowsIteration.append(self.matrix[i][j] * other.matrix[i][j])
                sol.append(rowsIteration)
            return Matrix(sol, self.row, self.column)
        
        else:
            print("Wrong Dimension")
           
     #multiplication operation between a matrix and integer/float constant     
    def matrixMultiplicationWithIntegerOrFloat(self, t):    
        if (type(t) == float or type(t) == int):
            sol = []
            for i in range(self.row):
                rowsIteration = []
                for j in range(self.column):
                    rowsIteration.append(self.matrix[i][j] * t)
                sol.append(rowsIteration)
            return Matrix(sol, self.row, self.column)
        else:
            print("The type of number you entered is neither float, nor integer")

    #transposing the matrix
    def matrixTranspose(self):
        sol = []
        for i in range(self.column):
            rowsIteration = []
            for j in range(self.row):
                rowsIteration.append(self.matrix[j][i]) #we reverse it because of transpose
            sol.append(rowsIteration)
        return Matrix(sol, self.column, self.row) 
    
        #copy of the matrix used in determinant function
    def new_matrixcopylist(self):
        #Make zeros matrix
        sol = [[0 for j in range(self.column)] for i in range(self.row)]
        
        #Put old values into zeroes matrix
        for i in range(self.row):
            for j in range(self.column):
                sol[i][j] = self.matrix[i][j]
        return sol #returns as list         
    
    def new_matrix_for_det(self, i):
        arr = self.new_matrixcopylist()
        if len(arr) == 2:
            return arr
        else:
            arr.pop(0)
            for j in arr:
                j.pop(i)
            return Matrix(arr, self.row, self.column)
    
    #reducing matrix to upper triangular form in order to calculate determinant easily
    def upper_triangle(self):
        copy = self.new_matrixcopylist()
        #move from first index to right through columns and find the main diagonal element (fd)
        for fd in range(self.row):
            for i in range(fd+1, self.row):
                #changing numbers that are close to zero in order to avoid zero division error
                if copy[fd][fd] == 0:
                    copy[fd][fd] = 1.0e-18
                crScalar = copy[i][fd] / copy[fd][fd] 
                
                #For each row below the row with fd in it, a scaler is created that is equal 
                #to (element in row that is in the same column as fd) divided by (fd)
                
                for j in range(self.row):
                    copy[i][j] = copy[i][j] - crScalar * copy[fd][j]
                    # goes to the next diagonal element and repeat the steps until 
                    #reducing the matrix to the upper triangular form
        return Matrix(copy, self.row, self.column)
    
    #the function that returns the product of diagonal elements in the upper triangular
    #formed matrix which also means the determinant. Only works with square matrices
    def determinant(self):
        copy = self.upper_triangle()
        iterable = copy.new_matrixcopylist()
        
        if self.row != self.column:
            print("This is not a square matrix, therefore there is no determinant for this matrix")
            
        else:
            product = 1
            for i in range(self.row):
                product *= iterable[i][i]
            return product

        
    
class Vector():
        def __init__(self, vector, dim): #Constructor
            self.vector = vector
            self.dim = dim
        
        def displayVector(self): #Returns the vector in the called index, written for control purposes
            print(self.vector)        
        
        def __add__(self, other) -> Vector: #Operator overloading for addition, adds two different vectors that have same dimensions
            if self.dim == other.dim:
                sol = []
                for i in range(self.dim):
                    sol.append(self.vector[i] + other.vector[i])
                return Vector(sol, self.dim)
            
            else:
                print("Vector dimensions are different, can't operate addition")
                
        
        def __sub__(self, other) -> Vector: #Operator overloading for subtraction, subtracts two different vectors that have same dimensions
            if self.dim == other.dim:
                sol = []
                for i in range(self.dim):
                    sol.append(self.vector[i] - other.vector[i])
                return Vector(sol, self.dim)
            else:
                print("Vector dimensions are different, can't operate addition")
                
            
        #returns a multiplication operation between a n dimensional vector and integer/float constant   
        def vectorMultiplicationWithFloatOrInteger(self, scalar:float or int)->Vector:
            sol = []
            for i in range(self.dim):
                sol.append(self.vector[i] * scalar)
            return Vector(sol, self.dim)
            
        
        #returns a cross product operation between 3 dimensional vectors
        def crossProduct(self, other)->Vector:
            if self.dim != 3 or other.dim != 3:
                print("This operation can only be done with 3 dimensional vectors")
                
            else:
                sol = []
                
                x = self.vector[1] * other.vector[2] - self.vector[2] * other.vector[1]
                y = -(self.vector[0] * other.vector[2]- self.vector[2] * other.vector[0])
                z = self.vector[0] * other.vector[1] - self.vector[1] * other.vector[0]
                
                sol.append(x)
                sol.append(y)
                sol.append(z)
                
                return Vector(sol, 3)
        
        #returns a vector that consists of inner product operation between two same dimensional vectors
        def innerProduct(self, other) -> Vector:
            if self.dim != other.dim:
                print("Dimensions are different for vectors, can't operate inner product")
            else:
                sol = [] 
                for i in range(self.dim):
                    sol.append(self.vector[i] * other.vector[i])
                return Vector(sol, self.dim)
            
        #creates a n dimensional vector full of 1 in order to solve linear systems of equations
        def createVectorForGaussianElimination(self, dim):
            vector = []
            
            for i in range(dim):
                a = []
                a.append(1)
                vector.append(a)    
            return vector

#Gaussian elimination method
def GaussianElimination(a, b):
    operationCount = 0 #this variable is defined in order to calculate how many iterations are done during this function 
    a = np.array(a, float)
    b = np.array(b, float)
    
    
    n = len(b)
    for k in range(n):
        if np.fabs(a[k,k]) < 1.0e-12:
            for i in range(k+1, n):
                if np.fabs(a[i,k]) > np.fabs(a[k,k]):
                    operationCount += 1
                    for j in range(k,n):
                        a[k,j], a[i,j] = a[i,j] , a[k,j]
                    b[k] ,b[i] = b[i], b[k]
                    operationCount += 1
                    break
        pivot = a[k][k]
        for j in range(k,n):
            a[k,j] /= pivot
            operationCount += 1
        b[k] /= pivot
        operationCount += 1
        for i in range(n):
            if i == k or a[i,k] == 0: continue
            factor = a[i,k]
            operationCount += 1
            for j in range(k,n):
                a[i,j] -= factor * a[k,j]
            b[i] -= factor * b[k]
            operationCount += 1
    print("Operation count for this operation is = ", operationCount)
    return b, a
    
    



x = [] 
r = readMatricesFromFile("C:/Users/efetu/.spyder-py3/Matrix_file_640.txt")
rs = readSolutions("C:/Users/efetu/.spyder-py3/sol_640.txt")

a = Matrix(r, 641, 641)
b = Vector.createVectorForGaussianElimination(x, 641)

A,B = GaussianElimination(r, b)

end = time()

executionTime = end - start #displays the execution time of this program
print("Execution time = ", executionTime, "\n")
print("Solution =\n ", A, "\n")
print("Our solutions\n", rs, "\n")


for i in range(len(A)):
    for j in range(len(rs)):
        if i == j:
            res = (float(A[i])) - int(rs[j])
        else:
            pass
    print("The difference between the actual solution and our solution in ", i, "th index is = ", res)

# print("Format = ", B) #prints the last received matrix after gaussian elimination


