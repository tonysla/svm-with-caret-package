# SVM-with-caret-package
Testing SVM models & trying to predict with diabetes data taken from kaggle.com. There are three SVM models below 
using 'kernlab', 'pROC' & 'e1071' package via 'caret' package. All three models use same trainControl but different methods, 'svmRadial', 'svmLinearWeights' & 'svmRadialWeights'.

## Libraries Used

    library(caret)
    library(kernlab)
    library(pROC)
    library(e1071)

## About repository
This is a demonstration on how to run `svm` with `caret` package in R. There are three different `svm` methods used, `svmRadial`, `svmLinearWeights` & `svmRadialWeights`. The methods use same `trainControl` parameters and then see which of these three methods performs better on kaggle.com data. All three methods done in here, can be executed without using `caret` package, but I think caret makes it easier and also provides a lot of different `resampling` methods and other parameters. 

