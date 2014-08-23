library(RCurl)
library(caret)
library(doParallel)
library(Hmisc)


#downloading files
URL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
x<- getURL(URL)
pml.training <- read.csv(textConnection(x),sep=",",na.strings = c("NA",""),header=TRUE)




URL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
y<- getURL(URL)
pml.testing <- read.csv(textConnection(y),sep=",",na.strings = c("NA",""),header=TRUE)





#reading local files

pml.training <- read.table("./pml-training.csv",sep=",",na.strings = c("NA",""),header=TRUE)
pml.testing  <- read.table("./pml-testing.csv",sep=",",na.strings = c("NA",""),header=TRUE)


#get 'classe' column fron training and save for later use
classeTraining<-subset(pml.training,select=c("classe"))



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








#create the partition for training 80% training, 20% validate

inTrain <- createDataPartition(pml.training2$classe, p=0.80, list=FALSE)
training <- pml.training2[inTrain,]
#create the partition for the validation
validation <- pml.training2[-inTrain,]


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


#print results
fit

#saving fit results
save(fit,file="fit.RData")


Sys.time() 
cvalidation <- train(validation$classe~.,data=validation, method="rf",trControl=trainControl(method = "cv"))
save(cvalidation,file="cvalidation.RData")
save(validation,file="validation.RData")
Sys.time() 

#stop multicore nodes
stopCluster(cl)

#print report of the CrossValidation
cvalidation$resample
cvalidation$results


#print the confusion matrix between training and validation
confusionMatrix(predict(cvalidation, newdata=validation), validation$classe)


#print cross validation predictions
cvalidation$finalModel

#remove unused colums from testing
testing<-pml.testing[ , -which(names(pml.testing) %nin% names(training))]


#predicting
test_prediction<-predict(fit, newdata=testing)
test_prediction
save(test_prediction,file="prediction.RData")
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(test_prediction)
