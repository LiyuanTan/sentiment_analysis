import numpy as np
import os
import subprocess
 
class getMatBin():
    def __init__(self):
        if not os.path.exists('matbin'):
            raise(Exception("no matbin folder!!!"))
 
        lsInfo = subprocess.Popen('ls matbin', shell=True, stdout = subprocess.PIPE).stdout.readlines()
 
        if len(lsInfo)==0:
            raise(Exception("no data files!"))
 
        print ('\n\n\n')
        loadedVars = []
        for eachStr in lsInfo:
            tempStrArray = eachStr.split('_')
            if len(tempStrArray)!=4:
                print ("error file format of %s" %(eachStr))
                continue
 
            varName = tempStrArray[0]
            varSize = tempStrArray[1].split('.')
            varSize.reverse()
            shapeArray = []
            for eachTerm in varSize:
                shapeArray.append(int(eachTerm))
 
            typeStr = tempStrArray[2]
            shapeArray[-2], shapeArray[-1] = shapeArray[-1], shapeArray[-2]
            exec('self.'+varName + '=' + 'np.fromfile(' +
                    '"./matbin/' + eachStr[0:-1] + '", dtype=np.' +
                    typeStr + ')')
            #pdb.set_trace()
            exec('self.'+varName + '=' + 'self.'+varName + '.reshape(' + 
                    str(shapeArray) + ')')
            print ('variable ' + varName + " loaded!!")
 
def toNpBin(var,varName):
    if not os.path.exists('npbin'):
        os.mkdir('npbin')
    #else:
        #os.system('rm ./npbin/*.npbin')
 
    typeStr = str(var.dtype)
    if typeStr=='float64':
        typeStr='double'
 
    shape = list(var.shape)
    #pdb.set_trace()
    if len(shape)==1:
        shapeArray=[shape[0],1]
    else:
        #shape.reverse()
        shapeArray=shape
        #shapeArray[0], shapeArray[1] = shapeArray[1], shapeArray[0]
        filename = './npbin/'+varName+'_'+str(shapeArray)[1:-1].replace(', ','-')+ \
                    '_' + typeStr + '_' + '.npbin'
        var.tofile(filename)
 
    print ('write ' + varName +' done!')