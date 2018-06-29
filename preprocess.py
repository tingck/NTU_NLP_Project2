import nltk
import pandas as pd
import numpy as np
import gensim
import pickle

np.random.seed(5)

#Mapping of the labels to integers
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

# maxSentenceLen = [0,0,0]
# labelsDistribution = FreqDist()

#Mapping of the distance between word_i and entity1,2
distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in range(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)

#load file as pandas dataframe
def load_train_data(path):
	with open(path, 'r') as f:
	    x = f.readlines()
	x = [line.strip() for line in x]
	label, e1, e2, sentence = [], [], [], []

	for i in range(0, len(x), 4):
	    sentence_i = x[i].split("\t")[1][1:-1]
	    label_i = labelsMapping[x[i+1]]
	    
	    sentence_i = sentence_i.replace("<e1>", " _e1_ ").replace("</e1>", "")
	    sentence_i = sentence_i.replace("<e2>", " _e2_ ").replace("</e2>", "")
	    tokens = nltk.word_tokenize(sentence_i)
	    e1_i = tokens.index("_e1_")
	    del tokens[e1_i]
	    e2_i = tokens.index("_e2_")
	    del tokens[e2_i]
	    
	    label.append(label_i)
	    e1.append(e1_i)
	    e2.append(e2_i)
	    sentence.append(tokens)

	data = pd.DataFrame.from_dict({'label':label, 'e1':e1, 'e2':e2, 'sentence':sentence})
	return data

def load_test_data(testpath, anspath):
	with open(testpath, 'r') as f:
	    x = f.readlines()
	x = [line.strip() for line in x]
	with open(anspath, 'r') as f:
	    y = f.readlines()
	y = [line.strip() for line in y]
	label, e1, e2, sentence = [], [], [], []

	for i in range(0, len(x)):
	    sentence_i = x[i].split("\t")[1][1:-1]
	    label_i = labelsMapping[y[i].split("\t")[1]]
	    
	    sentence_i = sentence_i.replace("<e1>", " _e1_ ").replace("</e1>", "")
	    sentence_i = sentence_i.replace("<e2>", " _e2_ ").replace("</e2>", "")
	    tokens = nltk.word_tokenize(sentence_i)
	    e1_i = tokens.index("_e1_")
	    del tokens[e1_i]
	    e2_i = tokens.index("_e2_")
	    del tokens[e2_i]
	    
	    label.append(label_i)
	    e1.append(e1_i)
	    e2.append(e2_i)
	    sentence.append(tokens)

	data = pd.DataFrame.from_dict({'label':label, 'e1':e1, 'e2':e2, 'sentence':sentence})
	return data

#find max sentence length in data set
def max_sentence_len(data):
	return max(len(l) for l in data['sentence'])

def load_embedding_deps(path):
	word2embeddings = {}

	with open(path, 'r') as f:
	    x = f.readlines()

	for line in x:
		split = line.strip().split(" ")

		vector = np.array([float(num) for num in split[1:]], dtype = 'float32')
		word2embeddings[split[0]] = vector
	        
	return word2embeddings

def load_embedding_google(path):
	model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
	return model

def create_embeddings(datalist, modelpath):
	all_words = []
	embeddings = []
	word2Idx = {}

	for data in datalist:
		all_words += [w.lower() for sent in data['sentence'] for w in sent]
	words_list = list(set(all_words))
	# word2Idx = {word: idx for idx,word in enumerate(words_list)}

	with open(modelpath, 'r') as f:
	    x = f.readlines()

	for line in x:
		split = line.strip().split(" ")

		if len(word2Idx) == 0: #Add padding+unknown
			word2Idx["PADDING"] = len(word2Idx)
			vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
			embeddings.append(vector)

			word2Idx["UNKNOWN"] = len(word2Idx)
			vector = np.random.uniform(-0.25, 0.25, len(split)-1)
			embeddings.append(vector)

		if split[0].lower() in words_list:
			vector = np.array([float(num) for num in split[1:]])
			embeddings.append(vector)
			word2Idx[split[0]] = len(word2Idx)

	embeddings = np.array(embeddings)

	return word2Idx, np.array(embeddings)
	
def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN"]

def create_matrices(data, word2Idx, maxSentenceLen=100):
	labels = np.array(data['label'], dtype='int32')
	positionMatrix1 = []
	positionMatrix2 = []
	tokenMatrix = []

	for index, row in data.iterrows():
		pos1 = row['e1']
		pos2 = row['e2']
		tokens = row['sentence']

		tokenIds = np.zeros(maxSentenceLen)
		positionValues1 = np.zeros(maxSentenceLen)
		positionValues2 = np.zeros(maxSentenceLen)
        
		for idx in range(0, min(maxSentenceLen, len(tokens))):
			tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)

			distance1 = idx - int(pos1)
			distance2 = idx - int(pos2)

			if distance1 in distanceMapping:
			    positionValues1[idx] = distanceMapping[distance1]
			elif distance1 <= minDistance:
			    positionValues1[idx] = distanceMapping['LowerMin']
			else:
			    positionValues1[idx] = distanceMapping['GreaterMax']
			    
			if distance2 in distanceMapping:
			    positionValues2[idx] = distanceMapping[distance2]
			elif distance2 <= minDistance:
			    positionValues2[idx] = distanceMapping['LowerMin']
			else:
			    positionValues2[idx] = distanceMapping['GreaterMax']

		tokenMatrix.append(tokenIds)
		positionMatrix1.append(positionValues1)
		positionMatrix2.append(positionValues2)

	return labels, np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32')

if __name__ == '__main__':
	train_data = load_train_data('TRAIN_FILE.txt')
	test_data = load_test_data('TEST_FILE.txt', 'answer_key.txt')
	maxSentenceLen = max(max_sentence_len(train_data),max_sentence_len(test_data))
	word2Idx, embeddings = create_embeddings([train_data,test_data], 'deps.words')
	with open('embeddings.pickle', 'wb') as handle:
		pickle.dump(embeddings, handle, -1)

    # Create token matrix
	train_set = create_matrices(train_data, word2Idx, maxSentenceLen)
	test_set = create_matrices(test_data, word2Idx, maxSentenceLen)
	with open('train_set.pickle', 'wb') as handle:
		pickle.dump(train_set, handle, -1)
	with open('test_set.pickle', 'wb') as handle:
		pickle.dump(test_set, handle, -1)

    