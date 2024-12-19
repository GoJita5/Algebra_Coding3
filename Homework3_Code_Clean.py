#---------------------Algebra and Coding Third Homework---------------------------#
    #|########Student: Touer Mohamed Elamine.
    #|##########Group: 4.
    #|##Academic Year: 2024/2025.
    #|######Professor: Rezki Chemlal.
#---------------------Algebra and Coding Third Homework---------------------------#

import numpy as np
import itertools as it
from time import time

#-----------------------------#SOLUTION#-----------------------------#

#___[I--SETUP]________________________________________###:
#Basic functions for creating the decoding algorithms.
#--------------------------------------------------------


  
def ham(r): #Basically creates the Hamming matrix
    vr2 = it.product([0,1], repeat=r)
    next(vr2)  #remove (0,0,...,0)
    h = np.fromiter(it.chain(*vr2), dtype=int) 
    h = np.transpose(h.reshape((2**r)-1,r)) 
    return h


def HammingCode1(r, reshape=True):
    n = (2**r)-1
    v = it.product([0,1], repeat=n)
    c = np.array([], dtype=int)
    h = ham(r)
    for x in v:
        y = np.array(x)
        z = h.dot(y) #This is y*(H)^T.
        z = z%2
        if (np.any(z) == 0): #Check if y*(H)^T is zero on Z/2Z.
            c = np.append(c,y)
    if reshape == True:
        return c.reshape(2**(n-r) ,n) #reshape c so that each row is a codeword
    else: #Or just leave it as one big row.
        return c


def StandardTable(r, reshape=1):
    #Creates the standard table of the Hamming code, assuming we want to correct only one error.
    n = (2**r)-1
    c = HammingCode1(r)
    S = np.array([c])
    eyedentity = np.eye(n, dtype=int) #These are all the possible coset leaders, with A[j-1] = (0,...,0,1,0,....,0), 1 is at the jth position.
    for i in eyedentity:
        S = np.append(S, c+i)%2
    if reshape == 1: #This reshapes S so that every row i of S is a_i + C, where a_i is a coset leader
        return S.reshape(n+1, n*(2**(n-r)))
    elif reshape == 2: #This reshapes S into a one-column table, so every row i of S is a vector of the space V(n,2)
        return S.reshape((n+1)*(2**(n-r)), n)
    else: #Leaves S as one big row of all the vectors in the space.
        return S


def bin(x:np.ndarray): #Takes an array of binary numbers and gives the decimal representation of it.
    a = 2**np.arange(np.size(x)-1, -1, -1)
    return np.dot(x, a)



#___[II--THE_SOLUTION]________________________________________###:
#The actual thing. The teacher asked for a,b,c, but I added d.
#-------------------------------------------------------------



def Stdecode(x:np.ndarray, r): #a) Standard Table decoding.
    n = (2**r)-1 #Size of x
    A = np.eye(n, dtype=int) #All coset leaders of weight 1 (i.e. a_j, j>=1)
    s = StandardTable(r, reshape=2)
    w = np.where(np.all(s==x, axis=1)) #Finding the row where x exists in the one-column table.

    v = w[0]

    j = v[0]//(2**(n-r)) #This yields the row in which x exist in the actual table.
    if j == 0:
        return x
    else:
        x = (x-A[j-1])%2
        return x


def Syndrome(x:np.ndarray, r): #b) Syndromes decoding.
    n = (2**r)-1
    h = ham(r)
    g = (h.dot(x))%2 #This is the syndrome of x.
    b = bin(g) #This is the decimal representation of g.
    if b == 0: #If it's 0, then no error has occurred, so x is a codeword.
        return x
    e = np.zeros(n, dtype=int) #If not, then an error has occurred in position b of x, to correct it, return x-e
    e[b-1] = 1 #where e is a vector equal to 1 at position b.
    return (x-e)%2


def Oneerrcorr(x:np.ndarray, r): #c) One error correction algorithm.
    n = (2**r)-1
    h = ham(r)
    g = (h.dot(x))%2
    b = bin(g)
    if b == 0:
        return x
    x[b-1] = 1-x[b-1] #Flip the bit at position b. Should be faster for big r.
    return x


def Nearestneighbor(x:np.ndarray, r): #d) Correcting the error in x using the closest codeword to x.
    C = HammingCode1(r)
    for c in C:
        y = np.sum((x-c)%2) #This is w(x-c) which equals d(x,c).
        if y == 0 or y == 1: #One of these two conditions must hold, because C is a perfect error correcting code.
            return c


#Note: In all of these functions, it shouldn't be necessary to insert r, I just didn't want to write n = np.size(x), and
     # I didn't want to import the log2 function to write r = int(log2(n+1)). I know I'm weird.
     # So the user is expected to insert x with length n = 2^r -1 but After all this work I don't care to write a condition
     # to force the user to do it.


#___[III--Testing]________________________________________###:
#I created a big function to compare each two functions.
#-------------------------------------------------------------


def bigtest(func1, func2, r, k=0, disp=False, appr=18):
    #Tests 2 of the decoding functions with some or all vectors of size n, and gives execution times.
    #Also rounds the times to 'appr' decimal places (appr stands for "approximate to"). If you wish to see each individual test,
    #set disp to True. 
    n = (2**r)-1
    f1time= []
    f2time= []
    if k==0: #If the number of vectors is not specified, tests all the vectors of the space.
        V = it.product([0,1], repeat=n)
    else: #If the number is specified, it tests k vectors of the space randomly.
        V = np.random.randint(2, size=(k, n))
    for x in V:
        if disp == True:
            print("----------------------")
            print("Test for: ", x)
        y = np.array(x)
        t0 = time()
        func1(y, r)
        t1 = time()
        func2(y, r)
        t2 = time()
        a = t1-t0
        b=t2-t1
        f1time.append(round(a, appr))
        f2time.append(round(b, appr))
        if disp == True:
            print(f"{func1.__name__}: {round(a, appr)} seconds")
            print(f"{func2.__name__}: {round(b, appr)} seconds")

    if disp == True:        
        print("\n**************************\n--------------------------\n**************************\n")
    print(f"total:\n\t{func1.__name__} times: {f1time}\n\t{func2.__name__} times: {f2time}")
    c = sum(f1time)/len(f1time)
    d = sum(f2time)/len(f2time)
    print(f"----------\nAverages:\n\t{func1.__name__} times: {round(c, appr)}\n\t{func2.__name__} times: {round(d, appr)}")



def actualtest(func1, r, k=0, testcount=1, appr=18):
    #Tests one of decoding functions t times with some or all vectors of size n, and gives execution times, their averages, and the
    #total average.
    #Also rounds the times to 'appr' decimal places. 
    n = (2**r)-1
    averages = []
    print("_________________________________________________________________\n")
    for i in range(testcount):
        test1 = []
        if k==0: #If the number of vectors is not specified, tests all the vectors of the space.
            V = it.product([0,1], repeat=n)
        else: #If the number is specified, it tests k vectors of the space randomly.
            V = np.random.randint(2, size=(k, n))
        for x in V:
            y = np.array(x)
            t0 = time()
            func1(y, r)
            t1 = time()
            a = t1-t0
            test1.append(round(a, appr))
        print(f"test{i+1} = {test1}\n----------------")
        avg= round(sum(test1)/len(test1), appr)
        averages.append(avg)
        print(f"avg{i+1} = {avg}")
        print("_________________________________________________________________\n")

    AVERAGE = round(sum(averages)/len(averages), appr)
    print("******************************************************************************")
    print(f"------------------------- AVERAGE = {AVERAGE} ---------------------------------")
    print("******************************************************************************")



#This is the code for plotting some results I collected.
'''
import matplotlib.pyplot as plt

# 1<=r<=5, the case of r=5 is considered as 960 seconds (16 mins), it's actually way greater than that.
Stdecodet = [0.000116, 0.000216, 0.001962, 0.538265, 960]
Nearestneighbort = [6.6e-05, 0.000183, 0.002344, 0.504369, 960]
# 7<=r<=24, the cases r<7 is uninteresting, it's almost always 0. and the case r=25 will ruin the plot.
Syndromet = [0.000141, 0.000212, 0.000497, 0.000762, 0.001688, 0.004396, 0.010376, 0.022112, 0.047225, 0.122978,
              0.247249, 0.533834, 1.032130, 2.847949, 5.292109, 11.906927, 30.307088, 86.058183]
Oneerrcorrt = [0.000116, 0.000249, 0.000315, 0.000715 ,0.001413, 0.003308, 0.007164, 0.025415, 0.035672, 0.091223,
                0.208885, 0.407726, 1.112938, 1.916405, 4.204347, 9.311131, 18.552687, 53.292143]

#Figure1: Stdecode vs Nearestneighbor.
plt.figure(figsize=(10, 6))
plt.plot(list(range(1, 5)), Stdecodet[0:4], label='Stdecode times', marker="o", color="black")
plt.plot(list(range(1, 5)), Nearestneighbort[0:4], label="Nearestneighbor times", marker='x', color="green")
plt.legend()
plt.grid(True)
plt.xlabel('r', loc='right')
plt.ylabel('Execution Time (Seconds)', loc='top')


#Figure2: Syndrome vs Oneerrcorr.
plt.figure(figsize=(10, 6))
plt.plot(list(range(7, 25)), Syndromet, label='Syndrome times', marker='d', color="red")
plt.plot(list(range(7, 25)), Oneerrcorrt, label="Oneerrcorr times", marker='*', color="yellow")
plt.legend()
plt.grid(True)
plt.xlabel('r', loc='right')
plt.ylabel('Execution Time (Seconds)', loc='top')

for i in range(0,6):
    Syndromet.insert(0, 0)
    Oneerrcorrt.insert(0, 0)
for i in range(0,19):
    Stdecodet.append(960)
    Nearestneighbort.append(960)


#Figure3: Stdecode vs Oneerrcorr.
plt.figure(figsize=(10, 6))
plt.plot(list(range(1, 25)), Stdecodet, label='Stdecode times', marker='o', color="black")
plt.plot(list(range(1, 25)), Oneerrcorrt, label="Oneerrcorr times", marker='*', color="yellow")
plt.legend()
plt.grid(True)
plt.xlabel('r', loc='right')
plt.ylabel('Execution Time (Seconds)', loc='top')




#Figure4: Everything.
plt.figure(figsize=(10, 6))
plt.plot(list(range(1, 25)), Stdecodet, label='Stdecode times', marker='o', color="black")
plt.plot(list(range(1, 25)), Oneerrcorrt, label="Oneerrcorr times", marker='*', color="yellow")
plt.plot(list(range(1, 25)), Syndromet, label='Syndrome times', marker='d', color="red")
plt.plot(list(range(1, 25)), Nearestneighbort, label="Nearestneighbor times", marker='x', color="green")

plt.legend()
plt.grid(True)
plt.xlabel('r', loc='right')
plt.ylabel('Execution Time (Seconds)', loc='top')


plt.show()
'''