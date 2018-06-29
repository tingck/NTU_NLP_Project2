from keras.models import load_model
import numpy as np
import pickle
from sklearn.metrics import f1_score

labelsMapping = {'Other':0, 
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}
labelsMapping_inv = {v:k for k,v in labelsMapping.items()}

MODEL_FILE = 'model.h5'

with open('pickles/test_set_google.pickle', 'rb') as handle:
    yTest, sentenceTest, positionTest1, positionTest2 = pickle.load(handle)

model = load_model(MODEL_FILE)
predict = model.predict([sentenceTest, positionTest1, positionTest2])
predict = np.argmax(predict, axis = 1)
predict_class = [labelsMapping_inv[i] for i in predict]
with open("answer_key.txt", "r") as f:
	ans = f.readlines()
y_test = [labelsMapping[a.strip().split("\t")[1]] for a in ans]


with open("predict.txt", "w", encoding = 'utf8') as ans_output:
	for i, pred in enumerate(predict_class):
		ans_output.write(str(8001+i) + '\t' + pred + '\n')

print('micro:', f1_score(yTest, predict, average='micro'))

