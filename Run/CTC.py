import itertools
import numpy as np
import math
import random
import os
import cPickle as pickle
import xml.etree.ElementTree as ET
from keras.preprocessing import sequence
from keras.optimizers import SGD,RMSprop
from keras.layers import Lambda, Dense, Activation, Flatten, Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU, LSTM
from keras.models import *
from keras.layers.core import *
from numpy import argmax

alphabet = ".,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 "

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
            

               # time=[float(stroke.attrib['end_time'])-float(stroke.attrib['start_time']),0,0]                             
                #time_offset= float(stroke.attrib['start_time'])
              
                points=0
                
                for point in stroke.findall('Point'):
                    xarray.append(float(point.attrib['x']))
                    yarray.append(float(point.attrib['y']))
                    tarray.append(float(point.attrib['time']))
                    parray.append(1)
                    points=points+1    
                    #points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset,float(point.attrib['time'])-time_offset])               
                op=op+points
                pointarray.append(op)
                parray[-1]=0
                parray[len(parray)-points]=0
            
            xnarray.append(xarray[0])
            ynarray.append(yarray[0])
            tnarray.append(tarray[0])

            for i,j in zip(pointarray[:],pointarray[1:]):
                if (i!=0):
                    xnarray.append(xarray[i]-xarray[i-1])
                    ynarray.append(yarray[i]-yarray[i-1])
                    tnarray.append(tarray[i]-tarray[i-1])

                for point in range (i+1,j):
                    xnarray.append(xarray[point]-xarray[i])
                    ynarray.append(yarray[point]-yarray[i])
                    tnarray.append(tarray[point]-tarray[i])
            
            result=zip(tnarray,xnarray,ynarray,parray)
            sequence.append(result)
            #print (sequence)
            #print (sequence[][3][0],sequence[0][0][1],sequence[0][0][2],sequence[0][0][3])
            return sequence
        
        
        # function to read each individual text line file
        def getAscii(filename, line_number):
            with open(filename, "r") as f:
                s = f.read()
            s = s[s.find("CSR"):]
            if len(s.split("\n")) > line_number+2:
                s = s.split("\n")[line_number+2]
                return s
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
        
        
        
        # build stroke database of every xml file inside iam database
        asciis = []
        xarray=[]
        lengths=[]
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
                lengths.append(len(ascii))
                asciis.append(ascii)
                strokes=convert_stroke_to_array(stroke)
                xarray.append(strokes)
        f = open(data_file,"wb")
        yencoded=[one_hot(s) for s in asciis]
        yarray = np.array(yencoded)
        #print (yarray)
        pickle.dump([xarray,yencoded], f, protocol=2)
        maxlength=max(lengths)
        f.close()  
        return xarray,yarray,maxlength

# Reverse translation of numerical classes back to characters
def decode(labels):
    #print (labels)
    ret = []
    for c in labels:
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
    #print (out,out.shape)
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1)) #return indexes of highest probability letters at each point/timestep
        out_best = [k for k, g in itertools.groupby(out_best)] #removes adjacent duplicate indexes
        outstr = decode(out_best)
        ret.append(outstr)
    return ret   
    
    
data_dir = "data"
data_file = os.path.join(data_dir, "strokes_training_data.cpkl")
stroke_dir = data_dir + "/lineStrokes-small"
ascii_dir = data_dir + "/ascii-small"

#if not (os.path.exists(data_file)) :
print("\tcreating training data cpkl file from raw source")
x_train,y_train,max_len=preprocess(stroke_dir, ascii_dir, data_file)
#print (y_train)
#pad sequences with keras
y_train = sequence.pad_sequences(y_train,padding='post')
x_train = sequence.pad_sequences(x_train,padding='post')
print (x_train.shape, y_train.shape)
#print (y_train)
size=x_train.shape[0]
trainable=True
inputs = Input(name='the_input', shape=x_train.shape[1:], dtype='float32')
rnn_encoded = Bidirectional(GRU(64, return_sequences=True),
                                name='bidirectional_1',
                                merge_mode='concat',trainable=trainable)(inputs)
birnn_encoded = Bidirectional(GRU(64, return_sequences=True),
                                name='bidirectional_2',
                                merge_mode='concat',trainable=trainable)(rnn_encoded)
#decoder=GRU(128,activation='softmax',recurrent_activation='hard_sigmoid', return_sequences=True)(rnn_encoded)
output = TimeDistributed(Dense(66, activation='softmax'))(birnn_encoded)


y_pred = Activation('softmax', name='softmax')(output)

labels = Input(name='the_labels', shape=[max_len], dtype='int32') #[no. of labels is equal to the maximum length of a line]
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


#model = Model(inputs=inputs, outputs=y_hat)
#Model(inputs=inputs, outputs=y_pred).summary()


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
print(inputs._keras_shape, labels._keras_shape,input_length._keras_shape,label_length._keras_shape)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#sgd = SGD(lr=0.2, decay=1e-2, momentum=0.9, nesterov=True, clipnorm=5)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='Adadelta')

absolute_max_string_len=max_len
blank_label=len(alphabet)+1

labels = np.ones([size, absolute_max_string_len])
input_length = np.zeros([size, 1])
label_length = np.zeros([size, 1])
source_str = []
#print (x_train.shape[0])
for i in range (x_train.shape[0]):
    labels[i, :] = y_train[i]
    input_length[i] = x_train.shape[1]
    label_length[i] =len(y_train[i])
    source_str.append('')
inputs_again = {'the_input': x_train,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
outputs = {'ctc': np.zeros([size])} 
model.fit(inputs_again, outputs, epochs=200,batch_size=25)
test_func = K.function([inputs], [y_pred])

letters=decode_batch(test_func,x_train)
#print (letters)
fi=open('output200.txt','w')
for letter in letters:
	fi.write("%s \n" % letter)
	
#p=model.predict(inputs_again, verbose=1)
#print ("model output probability values:",p[0])
#print (p)
#print  ("decoded model output:",decode(p[0]))
#print ("decoded original output:",decode(y_train[0]))
