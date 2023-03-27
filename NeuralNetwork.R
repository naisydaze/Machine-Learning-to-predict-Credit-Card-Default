setwd("~/Desktop/BC2407 BA2/Project")

library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(neuralnet)
library(data.table)
library(caTools)
library(caret)
library('fastDummies')

#Import data
banksimR.dt <- fread('banksimReduced.csv')
summary(banksimR.dt)

#Remove  ' '
banksimR.dt[, category := substring(category,2, nchar(category)-1)]
banksimR.dt[, age := substring(age,2, nchar(age)-1)]
banksimR.dt[, gender := substring(gender,2, nchar(gender)-1)]
banksimR.dt[, customer := substring(customer,2, nchar(customer)-1)]
banksimR.dt[, merchant := substring(merchant,2, nchar(merchant)-1)]
str(banksimR.dt)

#Scale continuous: amount and step
banksimR.dt$amount1 <- (banksimR.dt$amount - min(banksimR.dt$amount))/(max(banksimR.dt$amount)-min(banksimR.dt$amount))
banksimR.dt$step1 <- (banksimR.dt$step - min(banksimR.dt$step))/(max(banksimR.dt$step)-min(banksimR.dt$step))

#View summary of dataset
summary(banksimR.dt)

# Let's treat abortions as categorical and see what happens. Neuralnet cannot handle factors, unlike nnet. Thus need to manually create dummy variables.
banksim1 <- dummy_cols(banksimR.dt, select_columns=c('category', 'gender','customer', 'merchant','age'))
banksim1 <- banksim1[,`:=`(V1 = NULL, step = NULL, amount = NULL, customer=NULL,merchant=NULL, age=NULL,gender=NULL,category=NULL)]

colnames(banksim1)

col_list <- paste(c(colnames(banksim1)[colnames(banksim1) != "fraud"]),collapse="+")
col_list <- paste(c("fraud~",col_list),collapse="")
col_list

#create formula for nn
f <- formula(col_list)
f 

set.seed(2014)  # for random starting weights

# 30% trainset, 70% testset
train <- sample.split(Y=banksim1$fraud, SplitRatio = 0.3)
banksim.trainset <- subset(banksim1, train == T)
banksim.testset <- subset(banksim1, train == F)

str(banksim.trainset)
summary(banksim.testset$fraud)

####### Neural Network ######


m1 <- neuralnet(f, data=banksim.trainset, hidden=2, err.fct="ce", linear.output=FALSE, stepmax = 10000000)

par(mfrow=c(1,1))
plot(m1)

m1$net.result  # predicted outputs. 
m1$result.matrix 
m1$startweights
m1$weights

m1$generalized.weights

predict = compute(m1, banksim.testset)
predict$net.result

pred.m1 <- ifelse(unlist(m1$net.result) > 0.5, 1, 0)
pred.m1

cat('Trainset Confusion Matrix with neuralnet (1 hidden layer, 2 hidden nodes, Scaled X):')
table(banksim.trainset$fraud, pred.m1)

predict = compute(m1, banksim.testset)
prob <- predict$net.result

test.m1 <- ifelse(prob > 0.5, 1, 0)
cat('Testset Confusion Matrix with neuralnet')
t1<-table(actual=banksim.testset$fraud, predicted=test.m1)
t1

tn <- t1[1,1]
tp <- t1[2,2]
fp <- t1[1,2]
fn <- t1[2,1]

#False negative rate: fn/(fn+tp)
fnr <- fn/(tp+fn)
#False positive rate: fp/(fp+tn)
fpr <- fp/(fp+tn)
#overalL accuracy of model: (tn+tp) / sameple size
overall <- (tn+tp)/(tp+tn+fp+fn)

nn.matrix <- matrix(c(round(overall*100, 5), round(fpr*100, 5), round(fnr*100, 5)),ncol=1,byrow=3)
colnames(nn.matrix)<-c("%")
rownames(nn.matrix)<-c("Overall Accuracy:", "False Positive Rate:" ,"False Negative Rate:")
nn.matrix

##### Redesigned dataset ######

#Import data
banksim2<- fread('categorized_merchant_customer_fraud01.csv')
summary(banksim2)
str(banksim2)

#Scale continuous: amount and step
banksim2$amount1 <- (banksim2$amount - min(banksim2$amount))/(max(banksim2$amount)-min(banksim2$amount))
banksim2$step1 <- (banksim2$step - min(banksim2$step))/(max(banksim2$step)-min(banksim2$step))

#View summary of dataset
summary(banksim2)

#Drop insignificant columns
banksim2 <- banksim2[,`:=`(V9 = NULL, V10 = NULL, V11 = NULL)]

#Check levels 
levels(factor(banksim2$age))
levels(factor(banksim2$customer))
levels(factor(banksim2$merchant))
levels(factor(banksim2$category))
levels(factor(banksim2$fraud_category))
levels(factor(banksim2$gender))

#Create dummy variables
banksim2 <- dummy_cols(banksim2, select_columns=c('category', 'gender','customer', 'merchant','age'))

banksim2.dt <- banksim2[,`:=`(step = NULL, amount = NULL, customer=NULL,merchant=NULL, age=NULL,gender=NULL,category=NULL)]

colnames(banksim2.dt)

#Generate formula
col_list2 <- paste(c(colnames(banksim2.dt)[colnames(banksim2.dt) != "fraud"]),collapse="+")
col_list2 <- paste(c("fraud~",col_list2),collapse="")
col_list2

#Create formula for nn
f2 <- formula(col_list2)
f2

# 30% trainset, 70% testset
train <- sample.split(Y=banksim2.dt$fraud,SplitRatio = 0.3)
banksim2.trainset <- subset(banksim2.dt, train == T)
banksim2.testset <- subset(banksim2.dt, train == F)

summary(banksim2.trainset)

#Train neural network - 1 hidden layer with 3 nodes
m2 <- neuralnet(f2, data=banksim2.trainset, hidden=2, linear.output=FALSE, stepmax = 10000000)
plot(m2)

predict2 = compute(m2, banksim2.testset)
predict2$net.result

pred.m2 <- ifelse(unlist(m2$net.result) > 0.5, 1, 0)

cat('Trainset Confusion Matrix with neuralnet (1 hidden layer, 2 hidden nodes, Scaled X):')
train.t2 <- table(banksim2.trainset$fraud, pred.m2)
train.tn2 <- train.t2[1,1]
train.tp2 <- train.t2[2,2]
train.fp2 <- train.t2[1,2]
train.fn2 <- train.t2[2,1]

#False negative rate: fn/(fn+tp)
train.fnr2 <- train.fn2/(train.tp2+train.fn2)
#False positive rate: fp/(fp+tn)
train.fpr2 <- train.fp2/(train.fp2+train.tn2)
#overalL accuracy of model: (tn+tp) / sameple size
train.overall2 <- (train.tn2+train.tp2)/(train.tp2+train.tn2+train.fp2+train.fn2)

train.nn.matrix2 <- matrix(c(round(train.overall2*100, 5), round(train.fpr2*100, 5), round(train.fnr2*100, 5)),ncol=1,byrow=3)
colnames(train.nn.matrix2)<-c("%")
rownames(train.nn.matrix2)<-c("Overall Accuracy:", "False Positive Rate:" ,"False Negative Rate:")
train.nn.matrix2


predict2 = compute(m2, banksim2.testset)
prob2 <- predict2$net.result

test.m2 <- ifelse(prob2 > 0.5, 1, 0)
cat('Testset Confusion Matrix with neuralnet')
t2<-table(banksim2.testset$fraud, test.m2)
t2

tn2 <- t2[1,1]
tp2 <- t2[2,2]
fp2 <- t2[1,2]
fn2 <- t2[2,1]

#False negative rate: fn/(fn+tp)
fnr2 <- fn2/(tp2+fn2)
#False positive rate: fp/(fp+tn)
fpr2 <- fp2/(fp2+tn2)
#overalL accuracy of model: (tn+tp) / sameple size
overall2 <- (tn2+tp2)/(tp2+tn2+fp2+fn2)

nn.matrix2 <- matrix(c(round(overall2*100, 5), round(fpr2*100, 5), round(fnr2*100, 5)),ncol=1,byrow=3)
colnames(nn.matrix2)<-c("%")
rownames(nn.matrix2)<-c("Overall Accuracy:", "False Positive Rate:" ,"False Negative Rate:")
nn.matrix2





#End