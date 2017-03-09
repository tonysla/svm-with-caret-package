library(caret)
library(kernlab)
library(pROC)

# read the data
gData <- read.csv('GenDataCont1.csv')
# check for missing values
gData <- na.omit(gData)

# selecting only 91 variable and 'o1' variable 
# to be assigned as labels
d_f <- gData[c(1 : 91, 93)]

# transforming 'o1' variable into a factor &
# change the values from 1 & 2 into levels 0 & 1
d_f$o1 <- factor(ifelse(d_f$o1 == 1, "No", "Yes"))

# split data into train and test set
indices <- sample(2, nrow(d_f), replace = TRUE, prob = c(0.8, 0.2))
traiNx <- d_f[indices == 1,]
tesTy <- d_f[indices == 2,]

cvCont <- trainControl(method = "repeatedcv", repeats = 3, number = 5,
                       summaryFunction = twoClassSummary, classProbs = TRUE)

# set seed for reproducibility
set.seed(1)
svmTune <- train(x = traiNx[,-92],
                 y = traiNx[,92],
                 method = "svmRadial",
                 tuneLength = 4,
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = cvCont)

# use summary to check model
svmTune
summary(svmTune)
plot(svmTune, transform.x = log10, xlab = expression(log[10](gamma)), 
     ylab = "cost")
# predict test data using trained model
tesTy$pred <- predict(svmTune, newdata = tesTy[,-92], type = 'raw')
confusionMatrix(tesTy$o1, tesTy$pred)
# check accuracy
accuracy <- table(Actual = tesTy$o1, Pred = tesTy$pred)
accuracy
# accuracy for each group in percentage
round(prop.table(accuracy, 1), 3) * 100
addmargins(round(prop.table(accuracy, 1), 3) * 100)

# check which variables/columns are most important
varImpact <- varImp(svmTune, scale = FALSE)
varImpact
# plot top 20 most important variables
plot(varImpact, 20, main = "svm Radial")

# plot the actual/known result
plot(tesTy$o1, main = 'Actual')$actual
# plot matrix of predicted accuracy
plot(accuracy, main = 'Accuracy')
# plot predicted from the model
plot(tesTy$pred, main = 'Predicted')

#                 Pred
#        Actual     No   Yes   Sum
#            No   92.3   7.7 100.0
#            Yes  16.7  83.3 100.0
#            Sum 109.0  91.0 200.0

# call 'caret' package to run the matrix below
addmargins(round(prop.table(confusionMatrix(accuracy)$table) * 100))

# use the below function to clear environment
# rm(list=ls())
