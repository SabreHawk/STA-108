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

model1 = lm(Y~X1)
press2 = sum((model1$residuals / (1 - lm.influence(model1)$hat)) ^ 2)
