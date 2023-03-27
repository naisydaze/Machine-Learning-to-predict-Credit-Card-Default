#packages used:
library(ggplot2)
library(corrplot)
library(car)
library(caTools)
library(rpart)
library(data.table)
library(rpart.plot)
library(data.table)
library(psych)
library(pastecs)
library(rpart)				       
library(rattle)				
library(caret)	
options(scipen=100) 

set.seed(2014)
setwd("")

categorized_merchant_customer_fraud01 <- fread("categorized_merchant_customer_fraud01.csv", stringsAsFactors = T)
#factorise fraud column
categorized_merchant_customer_fraud01$fraud <- factor(categorized_merchant_customer_fraud01$fraud)

#see summary and dtype of each variable
summary(categorized_merchant_customer_fraud01)
str(categorized_merchant_customer_fraud01)

set.seed(2014)

#Train test split 
train <- sample.split(Y = categorized_merchant_customer_fraud01$fraud, SplitRatio = 0.7)
trainset <- subset(categorized_merchant_customer_fraud01, train == T)
testset <- subset(categorized_merchant_customer_fraud01, train == F)

#Grow tree
m1 <- rpart(fraud~., data=trainset, method="class", control = rpart.control(minsplit=2, cp=0))
summary(m1)

# plots the maximal tree and results
rpart.plot(m1, nn= T,main = "Maximal Tree in BCI before Covid-19")

# prints the maximal tree m1 onto the console
print(m1)

# prints out the pruning sequence and 10-fold CV errors, as a table
printcp(m1)

# display the pruning sequence and 10-fold CV errors, as a chart
plotcp(m1, main = "Subtrees in fraud")

# Compute min CVerror + 1SE in maximal tree cart1
CVerror.cap <- m1$cptable[which.min(m1$cptable[,"xerror"]), "xerror"] + m1$cptable[which.min(m1$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart1.
i <- 1; j<- 4
while (m1$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp1 <- ifelse(i > 1, sqrt(m1$cptable[i,1] * m1$cptable[i-1,1]), 1)

# Prune tree with optimal cp value
m2 <- prune(m1, cp = cp1)

# prints the pruned tree m2 onto the console
print(m2)

# prints out the pruning sequence and 10-fold CV errors, as a table
printcp(m2)

summary(m2)

#remove less important variables, rerun codes form line 35 to 72 again
testset$gender <-  NULL
testset$customer <-  NULL
testset$step <- NULL
testset$age <- NULL
trainset$gender <-  NULL
trainset$customer <-  NULL
trainset$step <- NULL 
trainset$age <- NULL 

# plots the tree m2 pruned using cp1.
rpart.plot(m2, nn= T, main = "Pruned Tree with min CP, after removing less important variables",tweak=1.3)

# Test CART model m2 predictions

cart.predict <- predict(m2, newdata = testset, type = "class")
results <- data.frame(testset, cart.predict)
#Test model

y.test.hat <- predict(m2, type = 'class', newdata = testset)

table1 <- table(observed=testset$fraud, predicted=y.test.hat)
table1
mean(testset$fraud == y.test.hat)

TN = table1[1,1]
TP = table1[2,2]
FP = table1[1,2]
FN = table1[2,1]

# Initialize table numbers
table2 <- data.frame('table' = 1:5)
# Specify row name
rownames(table2) <- c('Accuracy','TP Rate','FN Rate', 'TN Rate', 'FP Rate')

table2[1,1] <- mean(testset$fraud==y.test.hat) 
table2[2,1] <- TP/(TP+FN)
table2[3,1] <- FN/(TP+FN)
table2[4,1] <- TN/(TN+FP)
table2[5,1] <- FP/(TN+FP)

table2
