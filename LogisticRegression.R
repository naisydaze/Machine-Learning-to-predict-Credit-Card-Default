library(data.table)
library(MASS)
library(caTools)
library(car)
install.packages("AICcmodavg")
library(AICcmodavg)
library(ggplot2)
library(cowplot)

data = fread("categorized_merchant_customer_fraud01.csv", stringsAsFactors = T)
summary(data)

#Split train-test set
set.seed(2014)
train <- sample.split(Y = data$fraud, SplitRatio = 0.7)
d.trainset <- subset(data, train == T)
d.testset <- subset(data, train == F)

#Basic Model____________________________________________
data.m1 = glm(fraud~., family=binomial, data=d.trainset)
summary(data.m1)

#Accuracy - confusion matrix
predict.log = predict(data.m1, newdata=d.testset, type="response")
d.testset[,fraud2:=ifelse(fraud==0,"FALSE","TRUE")]

confu.matrix1 = table(d.testset$fraud2, predict.log>0.5)
confu.matrix1

#Accuracy - table
Acc_table1 = matrix(c(sum(diag(confu.matrix1))/sum(confu.matrix1)*100, confu.matrix1[1,2]/sum(confu.matrix1[1,2],confu.matrix1[1,1])*100,confu.matrix1[2,1]/sum(confu.matrix1[2,1],confu.matrix1[2,2])*100), byrow=TRUE)
colnames(Acc_table1) = c("")
rownames(Acc_table1) = c("Overall accuracy","FP rate", "FN rate")
Acc_table1


#Secondary Model____________________________________________
data.m2 = glm(fraud~step + age + category + amount, family=binomial, data=d.trainset)
summary(data.m2)

#Accuracy - confusion matrix
predict.log = predict(data.m2, newdata=d.testset, type="response")
d.testset[,fraud2:=ifelse(fraud==0,"FALSE","TRUE")]

confu.matrix2 = table(d.testset$fraud2, predict.log>0.5)
confu.matrix2

#Accuracy - table
Acc_table2 = matrix(c(sum(diag(confu.matrix2))/sum(confu.matrix2)*100, confu.matrix2[1,2]/sum(confu.matrix2[1,2],confu.matrix2[1,1])*100,confu.matrix2[2,1]/sum(confu.matrix2[2,1],confu.matrix2[2,2])*100), byrow=TRUE)
colnames(Acc_table2) = c("")
rownames(Acc_table2) = c("Overall accuracy","FP rate", "FN rate")
Acc_table2


#Plot Logistic Regression
predicted.data = data.frame(probability.of.fraud = data.m2$fitted.values,
                            fraud=d.trainset$fraud)
predicted.data = predicted.data[order(predicted.data$probability.of.fraud,
                                      decreasing=FALSE),]
predicted.data$rank = 1:nrow(predicted.data)

ggplot(data=predicted.data, aes(x=rank,y=probability.of.fraud)) + 
  geom_point(aes(color=fraud),alpha=1,shape=4,stroke=2) + xlab("Index") + 
  ylab("Predicted probability of transation being fraud") + 
  labs(title="Logistic Regression Model")