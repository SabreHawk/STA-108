library(leaps)

mydata = read.table("sel.txt", header = T)
n = dim(mydata)[1]

sel = regsubsets(Y~., data = mydata, nbest = 3)

#obtain the best model for each value of p
summary(sel)
SSE = summary(sel)$rss
n = dim(mydata)[1]
p = rowSums(summary(sel)$which)
AIC = n * log(SSE) + 2 * p
BIC = n * log(SSE) + p * log(n)
#obtain the cp for each best model
summary(sel)$cp

#obtain the rsquared for each best model
summary(sel)$rsq

#obtain the adjusted r squared for each best model
summary(sel)$adjr2

SSE = summary(sel)$rss
n = dim(mydata)[1]
p = rowSums(summary(sel)$which)
AIC = n * log(SSE) + 2 * p
BIC = n * log(SSE) + p * log(n)

model1 = lm(mydata$Y ~ mydata$X1)
press1 = sum((model1$residuals / (1 - lm.influence(model1)$hat)) ^ 2)

model2 = lm(mydata$Y ~ mydata$X2)
press2 = sum((model2$residuals / (1 - lm.influence(model2)$hat)) ^ 2)

model3 = lm(mydata$Y ~ mydata$X3)
press3 = sum((model3$residuals / (1 - lm.influence(model3)$hat)) ^ 2)

model12 = lm(mydata$Y ~ mydata$X1 + mydata$X2)
press12 = sum((model12$residuals / (1 - lm.influence(model12)$hat)) ^ 2)

model13 = lm(mydata$Y ~ mydata$X1 + mydata$X3)
press13 = sum((model13$residuals / (1 - lm.influence(model13)$hat)) ^ 2)

model23 = lm(mydata$Y ~ mydata$X2 + mydata$X3)
press23 = sum((model23$residuals / (1 - lm.influence(model23)$hat)) ^ 2)

model123 = lm(mydata$Y ~ mydata$X1 + mydata$X2 + mydata$X3)
press123 = sum((model123$residuals / (1 - lm.influence(model123)$hat)) ^ 2)

## stepwise regression

mydata = read.table("sel.txt", header = T)
n = dim(mydata)[1]

model0 = lm(Y ~ 1, data = mydata) #the model with no predictor variable
modelF = lm(Y ~ ., data = mydata)

# using AIC
step(model0, scope = list(lower = model0, upper = modelF), direction = "both")

step(model0, scope = list(lower = model0, upper = modelF), direction = "forward")

