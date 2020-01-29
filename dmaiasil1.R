# Author: Debora Maia Silva
# series of ML applications

library(dplyr)
library(tidyr)
library(mlr)
library(GoodmanKruskal)
library(corrplot)
library(car)
library(caret)
library(SciViews)
library(randomForest)
library(polycor)
library(psych)
library(ModelMetrics)
library(e1071)
library(LncFinder)
library(clusterSim)
library(MASS)
library(clusterGeneration)
library(devtools)

options(java.parameters = "-Xmx30000m")
library(bartMachine)
set_bart_machine_num_cores(4)


set.seed(1234)
#setwd("C:/Users/dmaiasil/Documents/Purdue Projects/Spring 2018/Modeling/Final")


#####READING DATA####
df <- read.csv("recs2009_public.csv", stringsAsFactors = F, header = T, sep = ",")
str(df)

#filtering for CA
df <- df %>%
  filter(REPORTABLE_DOMAIN == 26)

#look at this beautiful, no NAs dataset!
#sapply(df, function (x) sum(is.na(x)))
#df <- df %>% drop_na()

df <- subset(df, select=c("KWHSPH", "KWHCOL","KWHWTH","CUFEETNGSPH","CUFEETNGWTH","HDD65","CDD65","AIA_Zone",
                          "TOTROOMS","EQUIPM","FUELHEAT","EQMAMT","H2OTYPE1","FUELH2O", "WHEATSIZ","WHEATAGE", "H2OTYPE2", 
                          "FUELH2O2", "WHEATSIZ2", "AIRCOND", "COOLTYPE", "USECENAC", "NUMBERAC", "USEWWAC", "ELECAUX",
                          "NHSLDMEM", "TOTHSQFT", "TOTUCSQFT", "TYPEHUQ", "YEARMADERANGE", "DWASHUSE", "CWASHER"))

#we will have two target variables
#eletric space conditioning and gas space conditioning
#there are no duplicates
#unique(df)

#ELETRIC SOURCE 

####DATA ACROBATICS####
#only getting the variables used in the eletric end use model by EIA
df1 <- df[,c(1:3, 6:28)]
df1$SpaceCond <- rowSums(df[,c(1:3)])
df1 <- subset(df1, select=c(-1:-3))
df1 <- df1[,c(24,23,22,1:21)]

#checking what's going on in the dataset
# distribution of each variable
par(mfrow=c(6,4))
for (i in 1:24) {
  hist(df1[,i], xlab=names(df1)[i], main=names(df1)[i], col="blue", prob=T)
}
hist(df[,"EQUIPM"], breaks = c(-2,2:12,21), main="Type of main space heating equpiment used
     ",col="red", labels = c("Not Applicable","Steam or Hot Water System"))

#for the linear models, we need to convert the categorical variables into dummy variables
#that being said, the robust models like random forest that do not need dummy variables
#can deal with them. for consistency, we will use dummy variables set for everything
dummyCols <- c("AIA_Zone","TOTROOMS","EQUIPM","FUELHEAT","EQMAMT","H2OTYPE1","FUELH2O", "WHEATSIZ","WHEATAGE", "H2OTYPE2", 
               "FUELH2O2", "WHEATSIZ2", "AIRCOND", "COOLTYPE", "USECENAC", "NUMBERAC", "USEWWAC", "ELECAUX",
               "NHSLDMEM")
df1[dummyCols] <- lapply(df1[dummyCols], factor)
df1.d <- cbind(df1[,1:5], createDummyFeatures(df1, cols = dummyCols))
df1.d <- subset(df1.d, select = c(-1:-5))

#for MLR and SVM, we need to standardize the continuous variables
# use min-max transformation on continous variables

df1.dz <- data.Normalization(df1.d[,c(2:5)], type="n4")
df1.dz <- cbind(df1.d[,1], df1.dz, df1.d[,6:115])
colnames(df1.dz)[1] <- "SpaceCond"

##### PCA ####
#honestly, does not look good
#the idea was to try and make a profile of the data
pca <- as.matrix(hetcor(data=df1.d))

pca <- princomp(covmat=pca)

#we do not explain more then 29% of variance with only 10 PC
#but the biplot looks awful with more
#expected for dummy variables: we need more variables to explain what's happening
#but we see a clear profiling based on the variables - numerical together, house stats...
pca1 <- principal(df1.d
                  , nfactors = 10     # number of componets to extract
                  , rotate = "none"  # can specify different rotations
                  , scores = T       # find component scores or not
)
pca1$loadings
biplot.psych(pca1)


####CORRELATION####
#we have numerical and categorical features together, yuck
#the package GoodmanKruskal will help us with that

#categorical corplot
df.c <- df[,c(1:3, 6:28)]
df.c$SpaceCond <- rowSums(df.c[,c(1:3)])
df.c <- subset(df.c, select=c(-1:-3))
df.c <- df.c[,c(24,23,22,1:21)]
df.c[dummyCols] <- lapply(df.c[dummyCols], factor)
df.c <- df.c[,c(1, 6:24)]
corMatrix <- GKtauDataframe(df.c)
plot(corMatrix)

#numeric corplot
corMatrix1 <- cor(df.1[,1:5])
corMatrix1[upper.tri(corMatrix1)] <- 0
ids <- which(abs(corMatrix1) > 0.7, arr.ind = T)
correlated_variables <- data.frame(subset(reshape2::melt(corMatrix1), value > 0.7))
correlated_variables %>%
  filter(Var1 != Var2) %>%
  arrange(-value)


#####PARAMETERS SELECTION####
#let's start this party

##### MLR ####
mlr1<-lm(SpaceCond~., data = df1.dz)
summary(mlr1)

folds <- cut(seq(1,nrow(df1.d)),breaks=10,labels=FALSE)
step <- list()

rmse.train.mlr <- vector("numeric", 10)
rmse.test.mlr <- vector("numeric", 10)

rmse.train.mlr.ln <- vector("numeric", 10)
rmse.test.mlr.ln <- vector("numeric", 10)

for(i in 1:10){
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  train <- df1.d[testIndexes, ]
  test <- df1.d[-testIndexes, ]
  
  #select a formula based model by AIC
  null<-lm(SpaceCond~1, data=train)
  full<-lm(SpaceCond~., data=train)
  step[i] <- step(null, scope = list(upper=full), data=train, direction="both")
  
  
}

#boxcox and ln transformations worse -> we will use only mlr without transformations
for(i in 1:10){
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  train <- df1.d[testIndexes, ]
  test <- df1.d[-testIndexes, ]
  
mlr2 <- lm(formula = SpaceCond ~  FUELH2O.5+    TOTHSQFT +      CDD65 + FUELHEAT.5 + USECENAC.3+  AIA_Zone.4+ 
             USECENAC.2 +WHEATSIZ2.2  + EQUIPM..2  + TOTUCSQFT +  ELECAUX.0 + FUELH2O2.5 +  USEWWAC.2 +
             USEWWAC.3  +NHSLDMEM.2 +WHEATSIZ.3 +   EQUIPM.3 + TOTROOMS.8+  TOTROOMS.7 + TOTROOMS.6 +USECENAC..2 +
             EQUIPM.12 +TOTROOMS.12 +   EQMAMT.2 + AIA_Zone.3 + WHEATAGE.5+  H2OTYPE2.2, data = train)

model.pred.mlr <- predict(mlr2, test)
model.pred.OS.mlr <- predict(mlr2, test)

rmse.train.mlr[i] <- rmse(actual = train$SpaceCond, predicted = model.pred.mlr)
rmse.testmlr[i] <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.mlr)

train0 <- train %>% filter(!SpaceCond == 0)
test0 <- test %>% filter(!SpaceCond == 0)

BC<-boxcox(SpaceCond ~  FUELH2O.5+    TOTHSQFT +      CDD65 + FUELHEAT.5 + USECENAC.3+  AIA_Zone.4+ 
                USECENAC.2 +WHEATSIZ2.2  + EQUIPM..2  + TOTUCSQFT +  ELECAUX.0 + FUELH2O2.5 +  USEWWAC.2 +
                USEWWAC.3  +NHSLDMEM.2 +WHEATSIZ.3 +   EQUIPM.3 + TOTROOMS.8+  TOTROOMS.7 + TOTROOMS.6 +USECENAC..2 +
                EQUIPM.12 +TOTROOMS.12 +   EQMAMT.2 + AIA_Zone.3 + WHEATAGE.5+  H2OTYPE2.2, data = train0, lambda = seq(-4,4))
lamdamax<- BC$x[which.max(BC$y)]

mlr3 <- lm(formula = SpaceCond**(lamdamax) ~ FUELH2O.5+    TOTHSQFT +      CDD65 + FUELHEAT.5 + USECENAC.3+  AIA_Zone.4+ 
             USECENAC.2 +WHEATSIZ2.2  + EQUIPM..2  + TOTUCSQFT +  ELECAUX.0 + FUELH2O2.5 +  USEWWAC.2 +
             USEWWAC.3  +NHSLDMEM.2 +WHEATSIZ.3 +   EQUIPM.3 + TOTROOMS.8+  TOTROOMS.7 + TOTROOMS.6 +USECENAC..2 +
             EQUIPM.12 +TOTROOMS.12 +   EQMAMT.2 + AIA_Zone.3 + WHEATAGE.5+  H2OTYPE2.2, data = train0)

model.pred.mlr.ln <- predict(mlr3, test)
model.pred.OS.mlr.ln <- predict(mlr3, test)

rmse.train.mlr.ln[i] <- rmse(actual = train0$SpaceCond, predicted = model.pred.mlr.ln)
rmse.test.mlr.ln[i] <- rmse(actual = test0$SpaceCond, predicted = model.pred.OS.mlr.ln)
}



#gam did not deal well with dummies
#and had cringy worth results with the factors
#so let's go to the big guys 



##### RANDOM FOREST ####

folds <- cut(seq(1,nrow(df1.d)),breaks=10,labels=FALSE)

vi<- vector("character")

for(i in 1:10){
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  train <- as.matrix(df1.d[-testIndexes, ])
  
  rf.model<-randomForest(SpaceCond ~.,data=train,ntree=500,importance=T)
  varImpPlot(rf.model, sort = T, main="Variable Importance")
  
  temp <- as.data.frame(varImp(rf.model))
  temp$names <- colnames(train[,-1])
  temp <- temp[with(temp, order(-Overall)), ]
  temp <- temp[1:25,2]
  print(temp)

}

vars <- c("FUELHEAT.5",  "FUELH2O.5",   "CDD65",       "USEWWAC.1",   "USECENAC.1",  "TOTHSQFT",  
          "USECENAC..2", "AIRCOND.1" ,  "NUMBERAC.1"  ,"AIRCOND.0" ,  "EQUIPM.10",   "USECENAC.3" , 
          "COOLTYPE.2" , "USECENAC.2" , "COOLTYPE.1" , "ELECAUX.1"  , "FUELH2O.1" ,  "COOLTYPE..2" ,
          "FUELHEAT.1"  ,"TOTUCSQFT" ,  "HDD65"  ,     "EQUIPM.5" ,   "NUMBERAC..2", "USEWWAC.3"  , "USEWWAC..2" ,
          "FUELHEAT.5" , "FUELH2O.5" ,  "USECENAC.1" , "TOTHSQFT" ,   "USEWWAC.1"  , "NUMBERAC.1" , "CDD65"   ,    
          "USECENAC..2", "EQUIPM.10" ,  "COOLTYPE..2" ,"AIRCOND.0" ,  "FUELHEAT.1" , "COOLTYPE.1" , "EQUIPM.5",   
        "NUMBERAC..2" ,"USECENAC.2",  "TOTUCSQFT",   "USECENAC.3" , "USEWWAC..2" , "FUELH2O.1" ,  "AIRCOND.1"  , 
        "USEWWAC.3"  , "ELECAUX.0"  , "NUMBERAC.3" , "HDD65" ,  "FUELHEAT.5" , "TOTHSQFT"   , "FUELH2O.5" ,  "USEWWAC.1", 
        "CDD65"  ,     "USECENAC.1" , "USECENAC..2" ,"FUELHEAT.1" , "EQUIPM.10"  , "AIRCOND.1" ,  "COOLTYPE.2" , 
        "COOLTYPE..2", "COOLTYPE.1"  ,"USEWWAC.3" , "ELECAUX.1"  , "USECENAC.3" , "AIRCOND.0",   "EQUIPM.5" ,   "HDD65"
        ,  "FUELH2O.1" ,  "FUELH2O.2"  , "NUMBERAC..2" ,"FUELH2O2.5",  "TOTUCSQFT" ,  "USECENAC.2", 
"FUELHEAT.5" , "FUELH2O.5" ,  "TOTHSQFT"   , "USEWWAC.1" ,  "USECENAC.1",  "CDD65"      , "USECENAC..2" ,"ELECAUX.0",  
"COOLTYPE.1",  "AIRCOND.1"  , "ELECAUX.1"   ,"COOLTYPE..2" ,"NUMBERAC.1"  ,"FUELH2O.1" , 
 "USECENAC.2",  "EQUIPM.10" ,  "FUELHEAT.1" , "HDD65"      , "AIRCOND.0",   "COOLTYPE.2" , "USEWWAC.3"  , "EQUIPM.5",  "USECENAC.3",  "FUELH2O.2" ,  "NUMBERAC..2",
 "FUELHEAT.5" , "FUELH2O.5" ,  "TOTHSQFT"   , "USEWWAC.1",   "CDD65"     ,  "USECENAC.1",  "NUMBERAC.1" , "AIRCOND.1",   "ELECAUX.1"  , "EQUIPM.10"  , "FUELH2O.1"   ,"FUELHEAT.1",  "COOLTYPE..2", "COOLTYPE.1" ,
"FUELH2O2.5"  ,"USECENAC..2" ,"USECENAC.3"  ,"USEWWAC.3"  , "TOTUCSQFT" ,  "EQUIPM.5"   , "COOLTYPE.2"  ,"NUMBERAC.3" , "HDD65"       ,"USECENAC.2"  ,"NHSLDMEM.6" ,
 "FUELHEAT.5" , "FUELH2O.5"  , "USEWWAC.1"  , "TOTHSQFT"   , "USECENAC.1",  "CDD65"      , "NUMBERAC.1" , "COOLTYPE..2", "EQUIPM.10"  , "USECENAC..2", "AIRCOND.0"  , "ELECAUX.0"  , "USECENAC.3" , "TOTUCSQFT"  , "USECENAC.2" , "AIRCOND.1" ,  "ELECAUX.1",   "HDD65" ,      "COOLTYPE.2" , "USEWWAC.3"  , "FUELH2O2.5",  "COOLTYPE.1" , "FUELH2O.1"  , "NUMBERAC..2", "USEWWAC..2", 
 "FUELHEAT.5" , "USECENAC.1",  "FUELH2O.5"  , "CDD65"    ,   "USECENAC..2", "TOTHSQFT"  ,  "NUMBERAC.1" , "AIRCOND.1" ,  "USEWWAC.1"  , "COOLTYPE.1",  "NUMBERAC..2" ,"FUELHEAT.1" , "COOLTYPE..2" ,"USECENAC.3", 
 "USEWWAC.3"  , "HDD65"      , "USEWWAC..2" , "ELECAUX.0" ,  "FUELH2O.1"  , "USECENAC.2",  "EQUIPM.10"  , "EQUIPM.5"  ,  "AIRCOND.0"  , "COOLTYPE.2" , "FUELH2O2.5", 
 "FUELHEAT.5" , "TOTHSQFT"  ,  "FUELH2O.5"  , "USECENAC.1" , "USEWWAC.1"  , "USECENAC..2", "EQUIPM.10"  , "FUELHEAT.1",  "CDD65"      , "TOTUCSQFT" ,  "NUMBERAC.1" , "AIRCOND.1"  , "COOLTYPE.1",  "USECENAC.2" ,
 "USECENAC.3" , "AIRCOND.0"  , "ELECAUX.0"  , "COOLTYPE..2" ,"HDD65"      , "COOLTYPE.2",  "USEWWAC.3"  , "ELECAUX.1" ,  "NUMBERAC..2", "USEWWAC..2"  ,"EQUIPM.5"  , 
"FUELHEAT.5",  "FUELH2O.5",   "TOTHSQFT" ,   "USECENAC.1" , "USEWWAC.1"   ,"CDD65"      , "USECENAC..2", "AIRCOND.1"  , "NUMBERAC.1" , "FUELH2O.1"  , "EQUIPM.10" ,  "COOLTYPE..2" ,"AIRCOND.0" ,  "USECENAC.3" ,
"EQUIPM.5"  ,  "FUELHEAT.1",  "TOTUCSQFT",   "COOLTYPE.2" , "ELECAUX.0"   ,"NUMBERAC..2" ,"USECENAC.2",  "ELECAUX.1"  , "COOLTYPE.1",  "NUMBERAC.3"  ,"USEWWAC.3"  ,
"FUELHEAT.5" , "FUELH2O.5"  , "USEWWAC.1" ,  "CDD65"      , "TOTHSQFT"    ,"USECENAC.1"  ,"AIRCOND.1" ,  "TOTUCSQFT"  , "AIRCOND.0"  , "EQUIPM.10"  , "USECENAC..2" ,"USECENAC.3"  ,"FUELH2O.1"  , "NUMBERAC.1" ,
"NUMBERAC.3"  ,"HDD65"      , "FUELHEAT.1" , "COOLTYPE..2", "USEWWAC.3"   ,"COOLTYPE.2"  ,"ELECAUX.0"  , "FUELH2O2.5" , "COOLTYPE.1"  ,"EQUIPM.5"   , "USECENAC.2" )

#variables selected
rf.vars <- unique(vars)
rf.vars
##### SVM ####

folds <- cut(seq(1,nrow(df1.dz)),breaks=10,labels=FALSE)
rmse.train.svm <- vector("numeric", 10)
rmse.test.svm <- vector("numeric", 10)

rmse.kernel.train <- vector("numeric", 3)
rmse.kernel.test <- vector("numeric", 3)


#choosing kernel 
kernel <- c("linear","radial","polynomial")

#linear best choice
for(j in 1:3){
  
  for(i in 1:10){
    
    testIndexes <- which(folds==i,arr.ind=TRUE)
    train <- df1.dz[testIndexes, ]
    test <- df1.dz[-testIndexes, ]
    
    model.svm <- svm(SpaceCond ~., data = train, kernel = kernel[j])
    
    model.pred.svm <- predict(model.svm, test)
    model.pred.OS.svm <- predict(model.svm, test)
    
    rmse.train.svm[i] <- rmse(actual = train$SpaceCond, predicted = model.pred.svm)
    rmse.test.svm[i] <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.svm)
    
    
  }
  
  rmse.kernel.train[j] <- mean(rmse.train.svm)
  rmse.kernel.test[j] <- mean(rmse.test.svm)
  
}

#model with best kernel
model.svm <- svm(SpaceCond ~., data = train, kernel = "linear", scale=F)

#C and gamma parameters
#does CV implicitly -> we do it explicitly once our parameters are chosen
svm.tune <- tune(svm, SpaceCond~., data=train, kernel="linear", ranges=list(cost=10^(-2:2), gamma=2^(-2:2)), folds.num = 10)
#best parameters:  cost = 100  gamma = 0.25

svmProfile <- rfe(train, train$SpaceCond,
                  sizes = c(2, 5, 10, 20),
                  rfeControl = rfeControl(functions = caretFuncs,
                                          number = 200),
                  method = "svmLinear")


#  FUELH2O.5 FUELHEAT.5 FUELH2O.1  CDD65 FUELHEAT.1  AIA_Zone.5 USECENAC.3 USECENAC..2  COOLTYPE.1 HDD65 TOTHSQFT AIA_Zone.2 FUELH2O2.5 EQUIPM.4 TOTUCSQFT 

#### BartMachine ####
bart_machine_cv <- bartMachineCV(df1.d[,-1], df1.d$SpaceCond, serialize=T)
# bartMachine CV win: k: 5 nu, q: 10, 0.75 m: 200  

var_selection_by_permute_cv(bart_machine_cv, num_permute_samples = 10)
# "AIA_Zone.4"  "AIA_Zone.5"  "AIRCOND.0"   "CDD65"       "COOLTYPE..2" "FUELH2O.1"   "FUELH2O.5"   "FUELHEAT.1"  "FUELHEAT.5"  "NHSLDMEM.2" 
# "TOTHSQFT"    "TOTROOMS.8"  "USECENAC.2"  "USECENAC.3"  "WHEATSIZ.3"


####NNET####
intest <- createDataPartition(y = df1.dz$SpaceCond,
                               p = .70,            
                               list = FALSE)
train <- df1.dz[intest,]
test <- df1.dz[-intest,]


#regression type control
ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=10,       # k number of times to do k-fold
                     classProbs = F,  
                     summaryFunction = defaultSummary,
                     allowParallel=T
)

# Each model below is testing using different tuning parameters
myModel1 <-  train(SpaceCond ~ H2OTYPE2.2 +COOLTYPE.1+ USECENAC.2+ EQMAMT.3+ NUMBERAC.3+  EQUIPM.6 +TOTROOMS.10+  
                     FUELH2O2.5+ WHEATAGE.5+ TOTROOMS.4 +AIA_Zone.3 +EQUIPM.4+ TOTROOMS.2 + EQUIPM.5+
                     EQMAMT.2  +CDD65+ WHEATAGE.2 +USECENAC.3 +AIA_Zone.5+FUELH2O.5 +FUELHEAT.5,       
                   data = train[,c("SpaceCond","H2OTYPE2.2","COOLTYPE.1","USECENAC.2","EQMAMT.3","NUMBERAC.3","EQUIPM.6","TOTROOMS.10","FUELH2O2.5","WHEATAGE.5","TOTROOMS.4","AIA_Zone.3","EQUIPM.4","TOTROOMS.2","EQUIPM.5","EQMAMT.2","CDD65","WHEATAGE.2","USECENAC.3","AIA_Zone.5","FUELH2O.5","FUELHEAT.5")],     
                   method = "nnet",     
                   tuneLength = 1,
                   linout = )

myModel2 <- train(SpaceCond ~ .,       
                  data = train,     
                  method = "nnet",     
                  trControl = ctrl,    
                  tuneLength = c(1:3),
                  maxit = 100,
                  linout = 1
)  


myGrid <-  expand.grid(size = c(3, 5, 10, 20)    
                       , decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)) 
myModel3 <- train(SpaceCond ~ .,       
                  data = train,     
                  method = "nnet",     
                  trControl = ctrl,    
                  tuneGrid = myGrid,
                  linout = 1
)


myGrid <-  expand.grid(size = seq(from = 1, to = 10, by = 1)     
                       , decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7))  
myModel4 <- train(SpaceCond ~ .,       
                  data = train,     
                  method = "nnet",     
                  trControl = ctrl,    
                  tuneGrid = myGrid,
                  maxit = 500,
                  linout = 1
)

model.pred.1 <- predict(myModel1, test)
model.pred.OS.1 <- predict(myModel1, test)

rmse.train.1 <- rmse(actual = train$SpaceCond, predicted = model.pred.1)
rmse.test.1 <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.1)

model.pred.2 <- predict(myModel2, test)
model.pred.OS.2 <- predict(myModel2, test)

rmse.train.2 <- rmse(actual = train$SpaceCond, predicted = model.pred.2)
rmse.test.2 <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.2)


model.pred.3 <- predict(myModel3, test)
model.pred.OS.3 <- predict(myModel3, test)


rmse.train.3 <- rmse(actual = train$SpaceCond, predicted = model.pred.3)
rmse.test.3 <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.3)


model.pred.4 <- predict(myModel4, test)
model.pred.OS.4 <- predict(myModel4, test)

rmse.train.4 <- rmse(actual = train$SpaceCond, predicted = model.pred.4)
rmse.test.4 <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.4)

#we choose model3 as the best one
data.frame(rmse.train.1,rmse.train.2,rmse.train.3,rmse.train.4)
data.frame(rmse.test.1, rmse.test.2, rmse.test.3, rmse.test.4)


#import 'gar.fun' from Github
devtools::source_gist("6206737", filename = "gar_fun.r")

par(mar=c(3,4,1,1),family='serif')
nnet.r <- gar.fun(train$SpaceCond,myModel3)

#H2OTYPE2.2 COOLTYPE.1 USECENAC.2 EQMAMT.3 NUMBERAC.3  EQUIPM.6 TOTROOMS.10  
#FUELH2O2.5 WHEATAGE.5 TOTROOMS.4 AIA_Zone.3 EQUIPM.4 TOTROOMS.2  EQUIPM.5
#EQMAMT.2  CDD65 WHEATAGE.2 USECENAC.3 AIA_Zone.5  FUELH2O.5 FUELHEAT.5



##### MODELS FIT ####

rmse.train.base <- vector("numeric", 10)
rmse.test.base <- vector("numeric", 10)

rmse.train.mlr <- vector("numeric", 10)
rmse.test.mlr <- vector("numeric", 10)

rmse.train.rf <- vector("numeric", 10)
rmse.test.rf <- vector("numeric", 10)

rmse.train.nnet <- vector("numeric", 10)
rmse.test.nnet <- vector("numeric", 10)

rmse.train.b <- vector("numeric", 10)
rmse.test.b <- vector("numeric", 10)

rmse.train.svm <- vector("numeric", 10)
rmse.test.svm <- vector("numeric", 10)

#analysing in the for loop showed to be troublesome; so we will save the models and then get their stats, qqplots, actual vs predicted, etc
filename <- c("models1.RData", "models2.RData", "models3.RData", "models4.RData","models5.RData","models6.RData","models7.RData","models8.RData","models9.RData","models10.RData")

folds <- cut(seq(1,nrow(df1.d)),breaks=10,labels=FALSE)

for(i in 1:10){
  
  testIndexes <- which(folds==i,arr.ind=TRUE)
  train <- df1.d[-testIndexes, ]
  test <- df1.d[testIndexes, ]
  
  #stand for SVM
  train.z <- data.Normalization(train[,c(2:5)], type="n4")
  train.z <- cbind(train[,1], train.z, train[,6:115])
  colnames(train.z)[1] <- "SpaceCond"
  
  test.z <- data.Normalization(test[,c(2:5)], type="n4")
  test.z <- cbind(test[,1], test.z, test[,6:115])
  colnames(test.z)[1] <- "SpaceCond"
  
  #bart
 #train.covariates <- train[,-1]
  #test.covariates <- test[,-1]
  
  train.covariates <- train[,c("AIA_Zone.4","AIA_Zone.5","AIRCOND.0" ,"CDD65","COOLTYPE..2",
                               "FUELH2O.1",   "FUELH2O.5", "FUELHEAT.1",  "FUELHEAT.5" , "NHSLDMEM.2", 
                               "TOTHSQFT" ,   "TOTROOMS.8" , "USECENAC.2" , "USECENAC.3",  "WHEATSIZ.3")]
  
  test.covariates <- test[,c("AIA_Zone.4","AIA_Zone.5","AIRCOND.0" ,"CDD65","COOLTYPE..2",
                               "FUELH2O.1",   "FUELH2O.5", "FUELHEAT.1",  "FUELHEAT.5" , "NHSLDMEM.2", 
                             "TOTHSQFT" ,   "TOTROOMS.8" , "USECENAC.2" , "USECENAC.3",  "WHEATSIZ.3")]
  train.response <- train[,1]

  
  ####mean only
  mean.only <- lm(SpaceCond ~ 1, data=train)
  
  model.pred.base <- predict(mean.only, train)
  model.pred.OS.base <- predict(mean.only, test)
  
  rmse.train.base[i] <- rmse(actual = train$SpaceCond, predicted = model.pred.base)
  rmse.test.base[i] <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.base)
  
  ####MLR
  mlr <- lm(formula = SpaceCond ~  FUELH2O.5+    TOTHSQFT +      CDD65 + FUELHEAT.5 + USECENAC.3+  AIA_Zone.4+ 
              USECENAC.2 +WHEATSIZ2.2  + EQUIPM..2  + TOTUCSQFT +  ELECAUX.0 + FUELH2O2.5 +  USEWWAC.2 +
              USEWWAC.3  +NHSLDMEM.2 +WHEATSIZ.3 +   EQUIPM.3 + TOTROOMS.8+  TOTROOMS.7 + TOTROOMS.6 +USECENAC..2 +
              EQUIPM.12 +TOTROOMS.12 +   EQMAMT.2 + AIA_Zone.3 + WHEATAGE.5+  H2OTYPE2.2, data = train)
  
  model.pred.mlr <- predict(mlr, train)
  model.pred.OS.mlr <- predict(mlr, test)
  
  rmse.train.mlr[i] <- rmse(actual = train$SpaceCond, predicted = model.pred.mlr)
  rmse.test.mlr[i] <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.mlr)
  
  
  ####Ranfom forest
  train.rf <- as.matrix(train)
  test.rf <- as.matrix(test)
  
  model.rf <- randomForest(SpaceCond ~  FUELHEAT.5 + FUELH2O.5 +  CDD65 +USEWWAC.1 +  USECENAC.1 + TOTHSQFT + USECENAC..2 +AIRCOND.1 +  NUMBERAC.1 + AIRCOND.0 +  EQUIPM.10 +  USECENAC.3 + COOLTYPE.2 + USECENAC.2+  
                    COOLTYPE.1 + ELECAUX.1 +  FUELH2O.1 +  COOLTYPE..2 +FUELHEAT.1 + TOTUCSQFT +  HDD65 +EQUIPM.5 +NUMBERAC..2 +USEWWAC.3 +  USEWWAC..2 + ELECAUX.0 +  NUMBERAC.3 + FUELH2O.2 +
                         FUELH2O2.5 + NHSLDMEM.6, data=train.rf, ntree=500, importance=T)
  
  model.pred.rf <- predict(model.rf, train.rf)
  model.pred.OS.rf <- predict(model.rf, test.rf)
  
  rmse.train.rf[i] <- rmse(actual = train.rf[,1], predicted = model.pred.rf)
  rmse.test.rf[i] <- rmse(actual = test.rf[,1], predicted = model.pred.OS.rf)
  
  ####SVM
  
  model.svm <- svm(SpaceCond ~ ., data = train.z, kernel = "linear", cost = 100,  gamma = 0.25)
  
  model.pred.svm <- predict(model.svm, train.z)
  model.pred.OS.svm <- predict(model.svm, test.z)
  
  rmse.train.svm[i] <- rmse(actual = train.z$SpaceCond, predicted = model.pred.svm)
  rmse.test.svm[i] <- rmse(actual = test.z$SpaceCond, predicted = model.pred.OS.svm)
  
  ####NNET
  myGrid <-  expand.grid(size = c(3, 5, 10, 20)    
                         , decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)) 
  model.nnet <- train(SpaceCond ~ .,     
                    data = train.z,     
                    method = "nnet",     
                    tuneGrid = myGrid,
                    linout = 1
  )
  
  model.pred.nnet <- predict(model.svm, train.z)
  model.pred.OS.nnet <- predict(model.svm, test.z)
  
  rmse.train.nnet[i] <- rmse(actual = train.z$SpaceCond, predicted = model.pred.nnet)
  rmse.test.nnet[i] <- rmse(actual = test.z$SpaceCond, predicted = model.pred.OS.nnet)
  
  #####Bart
  
  model.bart <- bartMachine(X=train.covariates, y=train.response, k=5, q=0.75,num_trees=200, serialize = T )
  
  model.pred.b <- predict(model.bart, train.covariates)
  model.pred.OS.b <- predict(model.bart, test.covariates)
  
  rmse.train.b[i] <- rmse(actual = train$SpaceCond, predicted = model.pred.b)
  rmse.test.b[i] <- rmse(actual = test$SpaceCond, predicted = model.pred.OS.b)
  
  
  #to explore, get graphs and tons of stuff later
  save(list = c("mean.only","mlr", "model.rf", "model.svm", "model.nnet", "model.bart"), file = filename[i])
  #just some memory clean up because bart is a monster
  rm(model.bart)
}

data.frame(rmse.train.base,rmse.train.mlr,rmse.train.rf,rmse.train.svm, rmse.train.b)
data.frame(rmse.test.base,rmse.test.mlr,rmse.test.rf,rmse.test.svm, rmse.test.b)

data.frame(mean(rmse.train.base),mean(rmse.train.mlr),mean(rmse.train.rf),mean(rmse.train.svm), mean(rmse.train.nnet), mean(rmse.train.b))
data.frame(mean(rmse.test.base),mean(rmse.test.mlr),mean(rmse.test.rf),mean(rmse.test.svm), mean(rmse.test.nnet), mean(rmse.test.b))

#####BEST MODEL: BART####
load("models3.RData")

testIndexes <- which(folds==3,arr.ind=TRUE)
train <- df1.d[-testIndexes, ]
test <- df1.d[testIndexes, ]

train.covariates <- train[,c("AIA_Zone.4","AIA_Zone.5","AIRCOND.0" ,"CDD65","COOLTYPE..2",
                             "FUELH2O.1",   "FUELH2O.5", "FUELHEAT.1",  "FUELHEAT.5" , "NHSLDMEM.2", 
                             "TOTHSQFT" ,   "TOTROOMS.8" , "USECENAC.2" , "USECENAC.3",  "WHEATSIZ.3")]

test.covariates <- test[,c("AIA_Zone.4","AIA_Zone.5","AIRCOND.0" ,"CDD65","COOLTYPE..2",
                           "FUELH2O.1",   "FUELH2O.5", "FUELHEAT.1",  "FUELHEAT.5" , "NHSLDMEM.2", 
                           "TOTHSQFT" ,   "TOTROOMS.8" , "USECENAC.2" , "USECENAC.3",  "WHEATSIZ.3")]

model.pred.b <- predict(model.bart, train.covariates)
model.pred.OS.b <- predict(model.bart, test.covariates)

plot(model.pred.OS.b, test$SpaceCond, xlab = "predicted bart", ylab = "actual.test")
abline(a=0,b=1)

investigate_var_importance(model.bart,num_replicates_for_avg = 20)

check_bart_error_assumptions(model.bart)

# wilcox test for significance
shapiro.test(rmse.test.b)

#IMPROVEMENT OVER BASE MODEL - 40%
(mean(rmse.test.base) - mean(rmse.test.b))/mean(rmse.test.base)

finalmodel1 <- model.bart

save(finalmodel1, file="dmaiasil1.RData")
