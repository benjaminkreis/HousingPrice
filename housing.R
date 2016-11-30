#! /usr/bin/env Rscript

# Started with code from Christian Miller
# https://github.com/cmiller01/kaggle/tree/master/house-prices-advanced-regression-techniques

library(plyr)
library(dplyr)
library(readr)
library(ggplot2)
library(mlbench)
library(caret)
set.seed(2016)

# read training data, characters as factors is helpful
df <- read.csv(file='train.csv')

Hmisc::describe(df)
# Plot response variable--relatively skewed so normalize with log
hist(df$SalePrice)
hist(log(df$SalePrice))

df$LogSalePrice <- log(df$SalePrice)
prep_data <- function(df) {
  # preps data frame, mostly by dealing with NAs for now
  # transform sale price to logs

  # SubClass is more of a factor than numeric
  df$MSSubClass <- as.factor(df$MSSubClass)

  # Deal with some the NAs

  # LotFrontage (what does NA mean?) almost 18% of data
  # for now just assign mean value
  df$LotFrontage[is.na(df$LotFrontage)] <- mean(df$LotFrontage,na.rm=TRUE)

  # MasVnrArea & MasVnrType (masonry veneer type)
  table(df$Exterior1st,df$MasVnrType,useNA='always')
  # None type is most frequent with each exterior type w/masonry
  df$MasVnrType[is.na(df$MasVnrType)] <- "None"
  df$MasVnrArea[is.na(df$MasVnrArea)] <- 0

  # Electrical (NA?, only one missing)
  df$Electrical[is.na(df$Electrical)] <- "SBrkr"

  # For these variables NA means "no"
  # FireplaceQu - NA means no fireplace
  # GarageType & other - NA means no garage
  # Pool - NA means no pool
  # Fence - NA means no fence
  # MiscFeature - NA means no other feature
  # Alley - NA means no alley access
  # BsmtQuality & BsmtCond and BsmtExposure - NA means no basement
  # BsmtFinType1, etc.
  # add NA as "None" type
  na_to_none <- function(x) {
    x <- as.character(x)
    # code as special _none
    x[is.na(x)] <- "_none"
    as.factor(x)
  }
  na_vars <- c("Fence","Alley","PoolQC","FireplaceQu","MiscFeature",
               "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
               "GarageType","GarageFinish","GarageQual","GarageCond")
  other_vars <- names(df)[!(names(df) %in% na_vars)]
  tmp <- colwise(na_to_none,na_vars)(df)
  df <- bind_cols(subset(df,select=other_vars),tmp)

  # some dummies for HasGarage or HasBasement
  df$HasBasement <- df$BsmtQual != "_none"
  df$HasGarage <- df$GarageQual != "_none"
  # set garage year build to zero for now
  df$GarageYrBlt[is.na(df$GarageYrBlt)] <- 0
  df
}

df <- prep_data(df)
#End of data prep

# Look at correlation, numeric only
# Based on http://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
correlationMatrix <- cor(df[sapply(df, is.numeric)])
#print(correlationMatrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
print("Highly Correlated:")
print(highlyCorrelated)

# initial split on log price (evaluation is based on RMSE of log prices)
train_idx <- createDataPartition(df$LogSalePrice, p = 0.7, list=FALSE)
# drop the raw SalePrice column
train <- subset(df[train_idx,],select=-c(SalePrice,Id))
test <- subset(df[-train_idx,],select=-c(SalePrice,Id))

# try model trees
# model trees seems best
model_trees_fit <- train(LogSalePrice ~ ., data=train, method = "cubist", metric= "RMSE", trControl = trainControl(method = "cv",number=5))

# Look at feature importance, also based on caret link above
print("Importance:")
importance <- varImp(model_trees_fit, scale=FALSE)
print(importance)
jpeg('importance.jpg')
plot(importance)
dev.copy(png,'importance.png')
garbage <- dev.off

# Feature selection
rfe_train <- subset(train,select=-c(LogSalePrice))
rfe_y  <- t(subset(train,select=c(LogSalePrice)))
nrow(rfe_train)
length(rfe_y)
print("NCOLS:")
print(ncol(rfe_train))
colnames( rfe_train )
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
#results <- rfe(rfe_train[,1:79], rfe_[,80], sizes=c(1:79), rfeControl=control)
results <- rfe(rfe_train, rfe_y, sizes=c(1:81), rfeControl=control)
print(results)
predictors(results)
jpeg('feature_selection.jpg')
plot(results, type=c("g", "o"))
garbage <- dev.off

model_trees_pred <- predict(model_trees_fit,test)
postResample(pred=model_trees_pred,obs=test$LogSalePrice)


test <- read.csv(file='test.csv')
test <- prep_data(test)
# there are some additional NAs in test dataset, set to mode for categorical
# set to 0 for basements...
test$MSZoning[is.na(test$MSZoning)] <- "RL"
test$Utilities[is.na(test$Utilities)] <- "AllPub"
test$Exterior1st[is.na(test$Exterior1st)] <- "VinylSd"
test$Exterior2nd[is.na(test$Exterior2nd)] <- "VinylSd"

test$BsmtFinSF1[is.na(test$BsmtFinSF1)] <- 0
test$BsmtFinSF2[is.na(test$BsmtFinSF2)] <- 0
test$BsmtUnfSF[is.na(test$BsmtUnfSF)] <- 0
test$TotalBsmtSF[is.na(test$TotalBsmtSF)] <- 0
test$BsmtFullBath[is.na(test$BsmtFullBath)] <- 0
test$BsmtHalfBath[is.na(test$BsmtHalfBath)] <- 0

test$KitchenQual[is.na(test$KitchenQual)] <- "TA"
test$Functional[is.na(test$Functional)] <- "Typ"
test$GarageCars[is.na(test$GarageCars)] <- 0
test$GarageArea[is.na(test$GarageArea)] <- 0

test$SaleType[is.na(test$SaleType)] <- "WD"

# deal with new factor level as hack for now
test$MSSubClass[test$MSSubClass=="150"] <- "160"

# use model trees w/cross validation
train <- subset(df,select=-c(SalePrice,Id))
model_tree_fit <- train(LogSalePrice ~ ., data=train, method = "cubist", metric= "RMSE", trControl=trainControl(method = "cv",number=10))
model_tree_pred <- predict(model_tree_fit,test)

# prediction back to dollars
final <- exp(model_tree_pred)
submission <- data.frame(Id=test$Id,SalePrice=final)
write_csv(submission,path='submission_2.csv')
