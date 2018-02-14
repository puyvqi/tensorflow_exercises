import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image
import os

data_file="e:\\mintst\\train-images.idx3-ubyte"
data_buf=open(data_file,'rb').read()
index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>iiii',data_buf,index)
##print(magic,numImages,numRows,numColumns)
imagesize=numColumns*numRows
index+=struct.calcsize('>iiii')
print("offset:",index)
fmt_image='>'+str(imagesize)+'B'
images=np.empty((numImages,imagesize))
##print(struct.calcsize())
for i in range(numImages):
    if(i+1)%10000==0:
        print("已解析%d"%(i+1)+"张")
    images[i]=np.array(struct.unpack_from(fmt_image,data_buf,index))
    index+=struct.calcsize(fmt_image)
for i,image in enumerate(images):
    ima=image.reshape(28,28)
    #fig=plt.figure()
    #plw=fig.add_subplot(111)

    #plt.imshow(ima,cmap='gray')
    #plt.show()
    ima=Image.fromarray(ima)

    if ima.mode != 'RGB':
        ima= ima.convert('RGB')
    ima.save("e:\\mnist_image\\train_"+str(i)+".jpg")


