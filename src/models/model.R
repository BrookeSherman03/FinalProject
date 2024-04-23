#load libraries for modeling section
library(data.table)
library(caret)
library(Metrics)
library(xgboost)

#read in files
#read in the interim files and any other necessary files
train <- fread("./FinalProject/volume/data/interim/train.csv")
test <- fread("./FinalProject/volume/data/interim/test.csv")

#keep train_y values
train_y <- train$reddit

#run dummies
dummies <- dummyVars(reddit~ ., data = train)
saveRDS(dummies, "./FinalProject/volume/models/dummies")
traindum <- predict(dummies, newdata = train)
testdum <- predict(dummies, newdata = test)

#prep for xgboost
dtrain <- xgb.DMatrix(traindum,label=train_y)
dtest <- xgb.DMatrix(testdum)

#hyper parameter tuning
param <- list(  objective           = "multi:softprob",
                gamma               = 0.01,
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.1,
                max_depth           = 15,
                min_child_weight    = 1.5,
                subsample           = 0.8,
                colsample_bytree    = 1.0,
                tree_method = 'hist'
)

#cross validate
XGBm <- xgb.cv(params = param, nfold = 7, num_class = 11, nrounds = 10000, missing = NA, data = dtrain, print_every_n = 10, early_stopping_rounds = 25)

#save best iteration and train the actual model
best_ntrees <- unclass(XGBm)$best_iteration
watchlist <- list(train = dtrain)
XGBm <- xgb.train(params = param, nrounds = best_ntrees, num_class = 11, missing = NA, data = dtrain, watchlist = watchlist, print_every_n = 1)
saveRDS(XGBm, "./FinalProject/volume/models/XGBm")

#predict test and save as data frame with correct names
pred <- predict(XGBm, newdata = dtest)
mat <- matrix(data = pred, nrow = 25000, byrow = TRUE)
sub <- data.frame(mat)
names(sub) <- c("redditcars", "redditCFB", "redditCooking", "redditMachineLearning", "redditmagicTCG", "redditpolitics", "redditRealEstate", "redditscience", "redditStockMarket", "reddittravel", "redditvideogames")

#use example to readd id column
ex <- fread("./FinalProject/volume/data/raw/example_sub.csv")
sub$id <- ex$id

#write out submission
fwrite(sub, "./FinalProject/volume/data/processed/sub4.csv")
