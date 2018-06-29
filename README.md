# NTU_NLP_Project2
### Team: 13  
  
# Scripts:  
- 3 of all:  
  - preprocess.py: data preprocessing
  - train.py: CNN implementation
  - predict.py: testing set prediction

# Requirements:  
- Python 3.5  
- gensim 3.1.0 
- keras 2.0.8 
- tensorflow 1.3.0 
  
# Files:  
> All required files should be named as follows and be in the same folder as scripts:  
- TRAIN_FILE.txt  
- TEST_FILE.txt 
- dep.words
  - https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/ 
  
# Descriptions of scripts:  
- preprocess.py:  
  - Execution:  
  ```
    python3 preprocess.py  
  ```
  - Output:  
  ```
    embeddings.pickle, train_set.pickle, test_set.pickle
  ```
- train.py:  
  - Execution:  
  ```
    python3 train.py  
  ```
  - Output:  
  ```
    CNN model file
  ```
- predict.py:  
  - Execution:  
  ```
    python3 predict.py  
  ```
  - Output:  
  ```
    predict.txt
  ```