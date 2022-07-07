
test ="index.html" 

file = open(test, "r")    
data = file.read()
file.close()

file = open(test, "w")
newdata = data.replace("_static", "static")
file.write(newdata)
file.close()
