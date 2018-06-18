import numpy as np
import os
from HTK import HTKFile
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# same as the labels in answer.mlf
label_list = ["ling", "yi", "er", "san", "si", "wu", "liu", "qi", "ba", "jiu"]
labels = {}
MFCC = HTKFile()

# load testing data
test_x = []
input_lengths = []
filelist = os.listdir(os.getcwd() + "/MFCC/testing")
filelist.sort()
for filename in filelist:
    MFCC.load("MFCC/testing/" + filename)
    test_x.append(np.array(MFCC.data))
    input_lengths.append(len(np.array(MFCC.data)))

test_x = pad_sequences(test_x, dtype='float', padding='post')
input_lengths = np.array(input_lengths)

test_model = load_model("model.h5")
pred = test_model.predict(test_x)

# decode CTC and output answer
with open("result/result_nn.mlf", "w") as f:
    f.write("#!MLF!#\n")
    for i in range(len(filelist)):
        f.write("\"*/N" + filelist[i][1:-4] + ".rec\"\n")
        curr_out = 10
        for j in pred[i, :input_lengths[i]]:
            v = np.argmax(j)
            if v != curr_out and v != 10:
                f.write(label_list[v] + "\n")
            curr_out = v
        f.write(".\n")
f.close()

# calculate the accuracy using HTK HResults
os.system('HResults -e ??? sil -e ??? sp -I labels/answer.mlf lib/models_sp.lst \
           result/result_nn.mlf >> result/accuracy_nn')
