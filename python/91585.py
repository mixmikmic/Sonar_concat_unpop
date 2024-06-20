# # convert 2 to 3
# 

import os 

for dirPath, dirNames, fileNames in os.walk("."):
    print(dirPath)
    for f in fileNames: 
        
        file_name = os.path.join(dirPath, f)
        print(file_name)
        if file_name.endswith('.py'):
            get_ipython().system('2to3 -w {file_name}')


