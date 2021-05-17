- [NaiveBayes](#naivebayes)
  - [1 Problem](#1-problem)
  - [2 Specific Process](#2-specific-process)
    - [Step1. Read the tsv file](#step1-read-the-tsv-file)
    - [Step2. Build the word vector](#step2-build-the-word-vector)
    - [Step3. Calculate the TF-IDF value](#step3-calculate-the-tf-idf-value)
    - [Step4. Naive Bayes](#step4-naive-bayes)
    - [Step5. Prediction](#step5-prediction)
# NaiveBayes

## 1 Problem

Twitter comment classification by using Naive Bayes model.


## 2 Specific Process

### Step1. Read the tsv file

First, we use pandas to read the Train.tsv  file and set the header. We set the 'Tweet text' as the features and 'sentiment label' as the labels. Then we can apply the same method to read the Valid.tsv and Test.tsv file. Besides, we use some functions to do some statistics. The records count of three file is shown below.

| File          | Train.tsv | Valid.tsv | Test.tsv |
| ------------- | --------- | --------- | -------- |
| Records Count | 8005      | 1377      | 3155     |

Because the type of these values is pandas.series, in order to better deal with these data, we use list() function to convert series to list.



### Step2. Build the word vector

In order to have a better accuracy in validation dataset and test dataset, we need to build the word vector for these three file, which means we need to use a list to combine there three files' features together.

```python
train_text = train['Tweet text']
valid_text = valid['Tweet text']
test_text = test['Tweet text']
all_text = train_text + valid_text + test_text
```

In the following step, we can use the feature selection package to build the word vector according the words that appear.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer   
count_v0= CountVectorizer();  
counts_all = count_v0.fit_transform(all_text);
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train_text);   
print(repr(counts_train.shape))  #(8005, 29261)
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_valid = count_v2.fit_transform(valid_text);  
print(repr(counts_valid.shape))  #(1377, 29261)
count_v3 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v3.fit_transform(test_text);  
print(repr(counts_test.shape))	#(3155, 29261)
```

The output shows that the word vector has 29261 dimension after we combine three file's tweet text contents. It also shows that there are totally 29261 distinct words in the corpus.



### Step3. Calculate the TF-IDF value

We need to use a **criterion** to define the built word vector, that is to use tf-idf value. TF is short for term frequency, while idf is short for inverted document frequency, they can better represent the word importance by multiplying together. The process is shown below.

```python
tfidftransformer = TfidfTransformer();    
train_data = tfidftransformer.fit(counts_train).transform(counts_train);
valid_data = tfidftransformer.fit(counts_valid).transform(counts_valid); 
test_data = tfidftransformer.fit(counts_test).transform(counts_test);
```

Now, the elements in word vector are different tf-idf value, which means we can use these value to train the Naive Bayes model.



### Step4. Naive Bayes

We use 2 methods to implement the Naive Bayes.

The first method is training the model by using sklearn, which is used for check. We use the training dataset to train the Naive Bayes model, then use the model to do the prediction based on the validtation dataset. Finally, we can get the accuracy of the Naive Bayes model is : 0.5809731299927379

```python
# Applying the Naive Bayes model by using package
from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB(alpha = 1)   
clf.fit(x_train, y_train); 
preds = clf.predict(x_valid);
num = 0
preds = preds.tolist()
# print(preds)
for i in range(len(preds)):
    if preds[i] == y_valid[i]:
        num += 1
accuracy = num/len(preds)
print('The accuracy of Naive Bayes is:' + str(accuracy))
```

***

The second method is training the model by direct implement the Naive Bayes.  The formula of Bayes is

![](https://tva1.sinaimg.cn/large/008eGmZEgy1gplrr0t3usj307s02ijrh.jpg)

When we assume that the features are independent of each other, that is

![](https://tva1.sinaimg.cn/large/008eGmZEgy1gplrssmfjfj30ck021wel.jpg)

Then, the calculation formula of Naive Bayes algorithm is as follows

![](https://tva1.sinaimg.cn/large/008eGmZEgy1gplrswmcf2j30os02ldgh.jpg)

We can do some **optimizations** as follow

* Since the values of some feature attributes may be small, the P values of multiple features may be multiplied to approximately 0.
  So we can take log of both sides of this formula and turn multiplication into addition, to avoid the class multiplication problem.
* In order not to change the parameters too dramatically, Laplace smoothing is used.

The core code of Naive Bayes is shown below.

```python
def buildNaiveBayes(self, xTrain):
        yTrain = xTrain.iloc[:,-1]
        # Count the probability for each word vector dimension
        yTrainCounts = yTrain.value_counts()
        # uses Laplace Smoothing
        yTrainCounts = yTrainCounts.apply(lambda x : (x + 1) / (yTrain.size + yTrainCounts.size)) 
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature':{}}
        propNamesAll = xTrain.columns[:-1]
        allPropByFeature = {}
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index)
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                # Count the probability for each feature
                propClassSummary = propDatas.value_counts()
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        propClassSummary[propName] = 0
                Ni = len(allPropByFeature[nameFeature])
                # uses Laplace Smoothing
                propClassSummary = propClassSummary.apply(lambda x : (x + 1) / (propDatas.size + Ni))
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature

        return retModel
# Predict the labels by using Naive Bayes
    def predictBySeries(self, data):
        curMaxRate = None
        curClassSelect = None
        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])
            PFeature = infoModel['PFeature']
            
            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)
                if not propsRate:
                    continue
                # We use log addition to avoid multiplying small decimal numbers continuously, close to 0
                rate += np.log(propsRate.get(val, 0))
            if curMaxRate == None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass
```



### Step5. Prediction

We can use the trained Naive Bayes model to do the prediction. The input is the test dataset tweet text content, which is represented by different tf-idf value.

The output is ndarray type prediction result. In order to output a tsv file, we need to convert ndarray type into dataframe type.

```python
predictions = clf.predict(x_test)
# Convert the data type
df = pd.DataFrame(predictions)
df.to_csv('./prediction.tsv',header = None,index=False)
```

>  The prediction file (prediction.tsv) is in the folder for your reference~~

