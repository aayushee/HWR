try:
	import configparser
except:
	from six.moves import configparser
import numpy as np
import itertools
import math
import random
from math import sqrt
import gc
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import xml.etree.ElementTree as ET
from keras.preprocessing import sequence
from keras.optimizers import SGD,RMSprop,Adam,Adagrad,Adadelta
from keras.layers import Lambda, Dense, Activation, Flatten, Bidirectional, TimeDistributed
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.layers.recurrent import GRU, LSTM
from keras.models import *
from keras.layers.core import *
from numpy import argmax

alphabet = " abcdefghijklmnopqrstuvwxyz"
cfg=configparser.ConfigParser()
cfg.read('Config.cfg')

def preprocess(stroke_dir, ascii_dir, data_file):
        # create data file from raw xml files from iam handwriting source.
        print("\tparsing dataset...")
        
        # build the list of xml files
        filelist = []
        # Set the directory you want to start from
        rootDir = stroke_dir
        for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                filelist.append(dirName+"/"+fname)

        # function to read each individual xml file
        def getStrokes(filename):
            tree = ET.parse(filename)
            root = tree.getroot()
            sequence = []
            xnarray=[]
            ynarray=[]
            tnarray=[]
            xarray=[]
            yarray=[]
            tarray=[]
            parray=[]
            pointarray=[0]
            op=0
           
            for stroke in root[1].findall('Stroke'):
                         
                points=0
                
                for point in stroke.findall('Point'):
                    xarray.append(float(point.attrib['x']))
                    yarray.append(float(point.attrib['y']))
                    tarray.append(float(point.attrib['time']))
                    parray.append(1)
                    points=points+1    
                                  
                op=op+points
                pointarray.append(op)
                parray[-1]=0
                parray[len(parray)-points]=0
            
            xnarray.append(xarray[0])
            ynarray.append(yarray[0])
            tnarray.append(tarray[0])

            for i,j in zip(pointarray[:],pointarray[1:]):
                if (i!=0):
                    xnarray.append(abs(xarray[i]-xarray[i-1]))
                    ynarray.append(abs(yarray[i]-yarray[i-1]))
                    tnarray.append(abs(tarray[i]-tarray[i-1]))

                for point in range (i+1,j):
                    xnarray.append(abs(xarray[point]-xarray[i]))
                    ynarray.append(abs(yarray[point]-yarray[i]))
                    tnarray.append(abs(tarray[point]-tarray[i]))
            
	    xnarray[0]=0
	    ynarray[0]=0
       	    tnarray[0]=0
            result=zip(tnarray,xnarray,ynarray,parray)
            sequence.append(result)
            return sequence
            
    	def count_letters(str,counts):
		for c in str:
			if c in counts:
				counts[c]+=1
			else:
				counts[c]=1
		return counts   

# function to read each individual text line file
        def getAscii(filename, line_number):
            with open(filename, "r") as f:
                s = f.read()
            s = s[s.find("CSR"):]
            if len(s.split("\n")) > line_number+2:
                s = s.split("\n")[line_number+2]
                return s.rstrip()
            else:
                return ""

            
        def one_hot(s):
            ret = []
            for char in s:
                ret.append(alphabet.find(char))
            return ret
            
        
       
        
        def convert_stroke_to_array(stroke):
            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            print (n_point)
            stroke_data = np.zeros((n_point, 4), dtype=np.float32)
            counter=0
            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    #print (stroke[0])
                    stroke_data[counter, 0] = stroke[j][k][0]
                    stroke_data[counter, 1] = stroke[j][k][1]
                    stroke_data[counter, 2] = stroke[j][k][2]
                    stroke_data[counter, 3] = stroke[j][k][3]

                    counter += 1
            #print stroke_data
            return stroke_data
    
    
    	def normalize(x_train):
        	xdim=[]
        	ydim=[]
        	tdim=[]
        	pdim=[]
        	for i in range(len(x_train)):
            		for j in range(len(x_train[i])) :
                		xdim.append(x_train[i][j][0])
                		ydim.append(x_train[i][j][1])
                		tdim.append(x_train[i][j][2])
                		pdim.append(x_train[i][j][3])
        	mx = sum(xdim) / len(xdim)
       		sdx = sqrt(sum([(xi - mx) ** 2 for xi in xdim]) / len(xdim))
	        my = sum(ydim) / len(ydim)
        	sdy = sqrt(sum([(yi - my) ** 2 for yi in ydim]) / len(ydim))
        	mt = sum(tdim) / len(tdim)
        	sdt = sqrt(sum([(ti - mt) ** 2 for ti in tdim]) / len(tdim))
        	mp = sum(pdim) / len(pdim)
       		sdp = sqrt(sum([(pi - mp) ** 2 for pi in pdim]) / len(pdim))

        	print (mx,sdx,my,sdy,mt,sdt,mp,sdp)
        	#print (x_train[0][1])
        	for i in range(len(x_train)):
            		for j in range(len(x_train[i])) :
                		x_train[i][j][0]=(x_train[i][j][0]-mx)/sdx
                		x_train[i][j][1]=(x_train[i][j][1]-my)/sdy
                		x_train[i][j][2]=(x_train[i][j][2]-mt)/sdt
                		x_train[i][j][3]=(x_train[i][j][3]-mp)/sdp

        	return x_train

            
            
# build stroke database of every xml file inside iam database
        chars=set('\"%\'&()[]#+*-/!.,;:?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        asciis = []
        xarray=[]
        totlen=0
        for i in range(len(filelist)):
            if (filelist[i][-3:] == 'xml'):
                stroke_file = filelist[i]
                print ('processing '+stroke_file)
                stroke = getStrokes(stroke_file)
                ascii_file = stroke_file.replace("lineStrokes","ascii")[:-7] + ".txt"
                line_number = stroke_file[-6:-4]
                line_number = int(line_number) - 1
                ascii = getAscii(ascii_file, line_number)
                if any((c in chars)for c in ascii):
                    continue
		
                asciis.append(ascii)		
                strokes=convert_stroke_to_array(stroke)
                xarray.append(strokes)
        f = open(data_file,"wb")
	counts={}
	for s in asciis:
		count=count_letters(s,counts)
    	yencoded=[one_hot(s) for s in asciis]
    	yarray = np.array(yencoded)
   	print (count)
    	print (xarray[0][1])
    	xscaled=normalize(xarray)
    	print (xscaled[0][1])
    	pickle.dump([xscaled,yarray], f, protocol=2)
    	f.close()    
    
# Reverse translation of numerical classes back to characters
def decode(labels):
    #print (labels)
    ret = []
    #print (type(labels))
    lab=[int(x) for x in labels]
    for c in lab:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1)) #return indexes of highest probability letters at each point/timestep
	out_best = [k for k, g in itertools.groupby(out_best)] #removes adjacent duplicate indexes
        #print (out_best)
	outstr = decode(out_best)
        ret.append(outstr)
    return ret   
    
def load_preprocessed(data_file):
	f=open(data_file,"rb")
	[x_train,y_train]=pickle.load(f)
	f.close()
	print (len(x_train))
	return x_train,y_train
 
def generator(x_train,y_train,batch_size):
    max_len=y_train.shape[1]
    j=0
    #print (x_train.shape, y_train.shape)
    while True:   
        for cbatch in range(0, x_train.shape[0], batch_size):
	    j=j+1
	    x_batch=x_train[cbatch:(cbatch + batch_size),:,:]
            y_batch=y_train[cbatch:(cbatch + batch_size)]
            size=x_batch.shape[0]
            labels = np.ones([size, max_len])
            input_length = np.zeros([size, 1])
            label_length = np.zeros([size, 1])
            source_str = []
           # print (cbatch)
            for i in range (x_batch.shape[0]):
                labels[i, :] = y_batch[i]
                input_length[i] = x_batch.shape[1]
                label_length[i] =len(y_batch[i])
                source_str.append('')
            inputs_again = {'the_input': x_batch,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
            outputs = {'ctc': np.zeros([size])}
            yield(inputs_again,outputs)
            

stroke_dir = "data/lineStrokes-all"
ascii_dir = "data/ascii-all"
   
data_file = cfg.get('DEFAULT','Data File')
trainSize=int(cfg.get('DEFAULT', 'TrainSet Size'))
testSize=int(cfg.get('DEFAULT', 'TestSet Size'))
if not (os.path.exists(data_file)) :
	print("\tcreating training data cpkl file from raw source...")
	preprocess(stroke_dir, ascii_dir, data_file)
print ("loading preprocessed data...")
x_train1,y_train1=load_preprocessed(data_file)
print (len(x_train1))
y_train1= sequence.pad_sequences(y_train1,padding='post',dtype='float32')
x_train1= sequence.pad_sequences(x_train1,padding='post',dtype='float32')

x_train=x_train1[:trainSize]
y_train=y_train1[:trainSize]
x_test=x_train1[-testSize:]
y_test=y_train1[-testSize:]
assert not np.any(np.isnan(x_train))

#gc.collect()
print (x_train.shape,y_train.shape)
del x_train1
del y_train1
max_len=y_train.shape[1]
size=x_train.shape[0]
trainable=True

#print(y_train[0:11])
n1=int(cfg.get('NEURAL NETWORK', 'LSTM Layer1 Neurons'))
n2=int(cfg.get('NEURAL NETWORK', 'LSTM Layer2 Neurons'))
n3=int(cfg.get('NEURAL NETWORK', 'LSTM Layer3 Neurons'))
d1=int(cfg.get('NEURAL NETWORK', 'Dense Layer Neurons'))
init=cfg.get('NEURAL NETWORK', 'Kernel Initializer')
bias=cfg.get('NEURAL NETWORK','Bias Initializer')

inputs = Input(name='the_input', shape=x_train.shape[1:], dtype='float32')
rnn_encoded = Bidirectional(LSTM(n1, return_sequences=True,kernel_initializer=init,bias_initializer=bias),name='bidirectional_1',merge_mode='concat',trainable=trainable)(inputs)
birnn_encoded = Bidirectional(LSTM(n2, return_sequences=True,kernel_initializer=init,bias_initializer=bias),name='bidirectional_2',merge_mode='concat',trainable=trainable)(rnn_encoded)
trirnn_encoded=Bidirectional(LSTM(n3,return_sequences=True,kernel_initializer=init,bias_initializer=bias),name='bidirectional_3',merge_mode='concat',trainable=trainable)(birnn_encoded)
output = TimeDistributed(Dense(d1, name='dense',kernel_initializer=init,bias_initializer=bias))(trirnn_encoded)

y_pred = Activation('softmax', name='softmax')(output)
model=Model(inputs=inputs,outputs=y_pred)
print (model.summary())

labels = Input(name='the_labels', shape=[max_len], dtype='int32') #[no. of labels is equal to the maximum length of a line]
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

#Model(inputs=inputs, outputs=y_pred).summary()


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
print(inputs._keras_shape, labels._keras_shape,input_length._keras_shape,label_length._keras_shape)

batch_size=int(cfg.get('MODEL FITTING', 'Batch Size'))
lr=cfg.getfloat('OPTIMIZER', 'Learning Rate')
epoch=int(cfg.get('MODEL FITTING', 'Number of Epochs'))
steps=int(cfg.get('MODEL FITTING', 'Steps Per Epoch'))
outfile=cfg.get('OUTPUT', 'Output File')
mfile=cfg.get('OUTPUT', 'Model File')
ch=int(cfg.get('OPTIMIZER','Choice of Optimizer'))
fname=cfg.get('OUTPUT','Output Graph')
print (batch_size,lr,epoch,steps,outfile,mfile,init)

if ch==1:
	opt=Adam(lr=lr,clipnorm=1.)
elif ch==2:
	opt=Adagrad(lr=lr,clipnorm=1.)
elif ch==3:
	opt=Adadelta(lr=lr,clipnorm=1.)
elif ch==4:
	opt=RMSprop(lr=lr,clipnorm=1.)
else:
    opt=SGD(lr=lr, decay=1e-2, momentum=0.9, nesterov=True, clipnorm=0.5)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
for layer in model.layers:
	g=layer.get_config()
	h=layer.get_weights()
	print (g)
	print (h)
gc.collect()
#gc.collect()
my_generator = generator(x_train,y_train,batch_size)
hist=model.fit_generator(my_generator,epochs=epoch,steps_per_epoch=steps,shuffle=True,use_multiprocessing=False,workers=1)
#model.fit(x=x_train,y=y_train,batch_size=10,epochs=10)
model.save(mfile)
del model
gc.collect()
history_dict=hist.history
loss_values=history_dict['loss']
epochs=range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(fname)
gc.collect()

test_func = K.function([inputs], [y_pred])

letters=decode_batch(test_func,x_train[:25])
str1=[]
for d in y_train[:25]:
	str1.append(decode(d))
zipped=zip(str1,letters)
print (zipped)
fi=open(outfile,'w')
fi.write("%s \n" % zipped)

letters_t=decode_batch(test_func,x_test)
str2=[]
for i in y_test:
	str2.append(decode(i))
zipped2=zip(str2,letters_t)
print (zipped2)
fi.write("%s \n"% zipped2)
fi.close()
