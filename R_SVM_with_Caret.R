
# load packages
library(caret)
library(kernlab)
library(pROC)

# Testing SVM models & trying to predict with diabetes data 
# taken from kaggle.com. There are three SVM models below 
# using 'kernlab', 'pROC' & 'e1071' package via 'caret' package. 
# All three models use same trainControl 

# read the data
diabetes <- read.csv('diabetes.csv')
# set variable headers all to lower case letters
names(diabetes) <- tolower(names(diabetes))
# check the data
head(diabetes)
str(diabetes)

# check for corroletaion among variables
cor(diabetes)
# also plot a matrix of variables
plot(diabetes, pair = T)

# turn variavle 'outcome' as a dependet/respond variable 
# to factor & set '0' & '1' into 'No' & 'Yes'
diabetes$outcome <- factor(ifelse(diabetes$outcome == 0, 'No', 'Yes'))

# check the structure of the data
# make sure that variable 'outcome' is now a factor variable
str(diabetes)

# split data into train validation & test set (the ideal way)
# you can choose to just split the data into train and test set
indices <- sample(3, nrow(diabetes), replace = TRUE, prob = c(0.6, 0.2, 0.2))
train_x <- diabetes[indices == 1,]
valid_set <- diabetes[indices == 2,]
test_y <- diabetes[indices == 3,]

cvCont <- trainControl(method = "boot", number = 3,
                       summaryFunction = twoClassSummary, 
                       classProbs = TRUE)

# set seed for reproducibility
set.seed(1)

# now let's train the first model
model_one <- train(x = train_x[, -9],
                   y = train_x[, 9],
                   method = "svmRadial",
                   tuneLength = 3,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = cvCont)
# check model
model_one
summary(model_one)

plot(model_one, transform.x = log10, 
     xlab = expression(log[10](gamma)), 
     ylab = "cost")

# predict test data using trained model
valid_set$pred <- predict(model_one, newdata = valid_set[,-9], type = 'raw')
confusionMatrix(valid_set$outcome, valid_set$pred)

# requires 'e1071' package 
library(e1071)

# let's train the second model. It shares same trControl
# with first model, but method now is 'svmLinearWeights'.
model_two <- train(x = train_x[, -9],
                   y = train_x[, 9],
                   metric = "ROC",
                   tuneLength = 3,
                   trControl = cvCont,
                   method = "svmLinearWeights",
                   #tuneGrid = ex_gr,
                   preProc = c("center", "scale"))
# check model
model_two
summary(model_two)
plot(model_two, transform.x = log10, 
     xlab = expression(log[10](gamma)), 
     ylab = "cost")

# predict test data using trained model
valid_set$pred_2nd <- predict(model_two, newdata = valid_set[,-9], type = 'raw')
confusionMatrix(valid_set$outcome, valid_set$pred_2nd)

model_three <- train(x = train_x[, -9],
                     y = train_x[, 9],
                     metric = "ROC",
                     tuneLength = 3,
                     trControl = cvCont,
                     method = "svmRadialWeights",
                     preProc = c("center", "scale"))

# check the model
model_three
summary(model_three)
plot(model_three, transform.x = log10, 
     xlab = expression(log[10](gamma)), 
     ylab = "cost")

# predict test data using trained model
valid_set$pred_3rd <- predict(model_three, newdata = valid_set[,-9], type = 'raw')
confusionMatrix(valid_set$outcome, valid_set$pred_3rd)

# test best trained model on the test set.
# Usually models under perform on test set. 
test_pred <- predict(model_two, newdata = test_y[,-9], type = 'raw')
confusionMatrix(test_y$outcome, test_pred)

# check accuracy
accuracy <- table(Actual = test_y$outcome, Pred = test_pred)
accuracy

# accuracy for each group in percentage
addmargins(round(prop.table(accuracy, 1), 3) * 100)

# check which variables/columns are most important
varImpact <- varImp(model_two, scale = FALSE)
varImpact
# plot top 20 most important variables
plot(varImpact, 8, main = "svm Radial")

# plot the actual/known result
plot(test_y$outcome, main = 'Actual')$actual
# plot matrix of predicted accuracy
plot(accuracy, main = 'Accuracy')
# plot predicted from the model
plot(test_pred, main = 'Predicted')

# use the below function to clear environment
rm(list=ls())

