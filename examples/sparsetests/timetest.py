import subprocess, os

device = "turing"

def runParN():
    f = "../results/sparse/matvec_" + device +".csv"
    subprocess.call('rm ' + f, shell=True)

    Ns = [500,1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 50000, 60000]

    Ns = [str(n) for n in Ns] 

    subprocess.call('echo N, DenseTime, SparseTime  >> ' + f , shell=True)
    for n in Ns:
        for i in range(3):
            print("n:", n , "loop:", i, " of 3")
            subprocess.call('./examples/sparsetests/matVec ' + n + ' >>' + f, shell=True)

def runMaxEigN():
    f = "../results/sparse/powerIter__" + device +".csv"
    subprocess.call('rm ' + f, shell=True)

    Ns = [500,1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000, 50000]

    Ns = [str(n) for n in Ns] 

    subprocess.call('echo N, DenseTime, SparseTime  >> ' + f , shell=True)
    for n in Ns:
        for i in range(3):
            print("n:", n , "loop:", i, " of 3")
            subprocess.call('./examples/sparsetests/matVec ' + n + ' >>' + f, shell=True)



runParN()
#runMaxEigN()
