import os 
import sys 

def rename(path):
    for f in path:
        print(f)
        if( (f == ".") or  (f == "/") or (f == "./") or (f == "/.") or (f == "..")):
            continue
        if(os.path.isdir(f)):
            rename(path +"/" + f)
        if(".html" in f):
            file = open(f, "r")    
            data = file.read()
            file.close()

            file = open(test, "w")
            newdata = data.replace("_static", "static")
            file.write(newdata)
            file.close()


rename("./")
