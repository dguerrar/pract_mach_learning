Practical Machine Learning - Personal Activity Data
========================================================


## Summary

Using devices such as Jawbone Up, Nike FuelBand, andFitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


## Data Processing

```r
#load data in memory from previous execution
load(file = "./fit.RData")
load(file = "./cvalidation.RData")
load(file = "./validation.RData")
load(file = "./prediction.RData")
```


Fist, load the required libraries:

```r
#for downloading the files
library(RCurl)
#for training 
library(caret)
#for parallel computing
library(doParallel)
#miscelaneous
library(Hmisc)
library(randomForest)
```


The second step must be downloading the data files from its sources:

```r
#downloading files
URL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
x<- getURL(URL)
pml.training <- read.csv(textConnection(x),sep=",",na.strings = c("NA",""),header=TRUE)




URL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
y<- getURL(URL)
pml.testing <- read.csv(textConnection(y),sep=",",na.strings = c("NA",""),header=TRUE)
```



## Exploratory Data Analysis
The wotk to do is to analize the 'training' and set make a preduction for the attribute 'classe' that is not present at the 'testing' set.
A 'validate' must be done by splitting the 'training' set at 80% and 20%.



```r
#get 'classe' column fron training and save for later use
classeTraining<-subset(pml.training,select=c("classe"))
```

There are 160 columns at the 'training' set. I only want the numeric ones, without NA values at any cells.
Finally, the Covariance is analizes so only variables != 0 must be at the final 'training' set.


```r
#delete NON numeric colums
pml.training2<-pml.training[sapply(pml.training,is.numeric)]

#setting repeatability
set.seed(12345)

#replacing NA values by 0.
pml.training2[is.na(pml.training2)] <- 0

#delete cero covariates
nzv <- nearZeroVar(pml.training2)
pml.training2<-pml.training2[-nzv]
pml.training2<-cbind(classeTraining,pml.training2)
```

Now is time to create the 'training' and the 'validate' set: 80%/20% each:


```r
#create the partition for training 80% training, 20% validate

inTrain <- createDataPartition(pml.training2$classe, p=0.80, list=FALSE)
training <- pml.training2[inTrain,]
#create the partition for the validation
validation <- pml.training2[-inTrain,]
```



Train the 'training' set with ***caret***. 'Training' set is pre-processed with the 'PCA' algorithm in order to reduce the columns to work in the train.


```r
#set multiples cpu cores to run
cl <- makeCluster(4)
registerDoParallel(4)
#getting num of nodes- for testing only
getDoParWorkers()

Sys.time() 
#preprocess, only want principal components, all others columns out.
#radom tree
fit<-train(classe ~ .,preProcess="pca",data=training,method="rf",importance=TRUE)
Sys.time() 
```


Train the 'validate' set with ***Cross Validation***. No preprocess method must be done.


```r
Sys.time() 
cvalidation <- train(validation$classe~.,data=validation, method="rf",trControl=trainControl(method = "cv"))
save(cvalidation,file="cvalidation.RData")
Sys.time() 


#stop the cluster
stopCluster(cl)
```

Now is time to do the testing with the 'testing' set. No preprocess method must be done.


```r
#remove unused colums from testing
testing<-pml.testing[ , -which(names(pml.testing) %nin% names(training))]
test_prediction<-predict(fit, newdata=testing)
```



## Results

First the Results from the train:


```r
fit
```

```
## Random Forest 
## 
## 15699 samples
##    56 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction, scaled, centered 
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 15699, 15699, 15699, 15699, 15699, 15699, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.003   
##   30    1         1      0.005        0.007   
##   60    1         1      0.005        0.007   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.78%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4461    1    0    0    2    0.000672
## B   18 3000   20    0    0    0.012508
## C    0   18 2710   10    0    0.010226
## D    0    1   42 2525    5    0.018655
## E    0    0    0    5 2881    0.001733
```

```r
fit$preProcess
```

```
## 
## Call:
## "scrubed"
## 
## Created from 15699 samples and 56 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 27 components to capture 95 percent of the variance
```

The pre-process get 27 components from the data set in order to reduce the columns to train.



Results of the Cross Validation and the Confussion Matrix:

```r
cvalidation$resample
```

```
##    Accuracy Kappa Resample
## 1         1     1   Fold02
## 2         1     1   Fold01
## 3         1     1   Fold04
## 4         1     1   Fold03
## 5         1     1   Fold06
## 6         1     1   Fold05
## 7         1     1   Fold08
## 8         1     1   Fold07
## 9         1     1   Fold10
## 10        1     1   Fold09
```

```r
cvalidation$results
```

```
##   mtry Accuracy  Kappa AccuracySD  KappaSD
## 1    2   0.9880 0.9848  0.0046601 0.005895
## 2   29   1.0000 1.0000  0.0000000 0.000000
## 3   56   0.9997 0.9997  0.0008026 0.001015
```

```r
#print the confusion matrix between training and validation
confusionMatrix(predict(cvalidation, newdata=validation), validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  759    0    0    0
##          C    0    0  684    0    0
##          D    0    0    0  643    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
#print cross validation predictions
cvalidation$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 29
## 
##         OOB estimate of  error rate: 0%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1116   0   0   0   0           0
## B    0 759   0   0   0           0
## C    0   0 684   0   0           0
## D    0   0   0 643   0           0
## E    0   0   0   0 721           0
```

As you can see on the Validate Model, **OOB estimate of  error rate: 0.78%** 
and  **Accuracy : 1** on **95% CI : (0.9991, 1)**

With that info, I can expect a correct prediction in the testing set.




Our predicion for the 20 records at the 'testing' set:

```r
test_prediction
```

```
##  [1] B A A A A A A B A A A A B A E B A B B B
## Levels: A B C D E
```




```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(test_prediction)
```


