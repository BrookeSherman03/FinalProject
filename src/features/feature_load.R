#load libraries
library(data.table)
library(Rtsne)

#read in data
train <- fread("./FinalProject/volume/data/raw/kaggle_train.csv")
test <- fread("./FinalProject/volume/data/raw/kaggle_test.csv")
train_emb <- fread("./FinalProject/volume/data/raw/train_emb.csv")
test_emb <- fread("./FinalProject/volume/data/raw/test_emb.csv")

#change reddit variables to numerals
train$reddit[train$reddit == "cars"] <- 0
train$reddit[train$reddit == "CFB"] <- 1
train$reddit[train$reddit == "Cooking"] <- 2
train$reddit[train$reddit == "MachineLearning"] <- 3
train$reddit[train$reddit == "magicTCG"] <- 4
train$reddit[train$reddit == "politics"] <- 5
train$reddit[train$reddit == "RealEstate"] <- 6
train$reddit[train$reddit == "science"] <- 7
train$reddit[train$reddit == "StockMarket"] <- 8
train$reddit[train$reddit == "travel"] <- 9
train$reddit[train$reddit == "videogames"] <- 10

#prep train and test for a master table
train_emb$reddit <- train$reddit
train_emb$id <- train$id
test_emb$reddit <- 11 #will make it easier to split back to train and test from master
test_emb$id <- test$id

#bind tables together
master <- rbind(train_emb, test_emb)

#run pca and tsne on data
v_data <- data.frame(master[,0:512]) #only take the v1-v512 columns for pca
pca <- prcomp(v_data)
pca_dt <- data.table(unclass(pca)$x)
tsne <- Rtsne(pca_dt, pca = F, dims = 3, perplexity = 70, check_duplicates = F, max_iter = 500) #this takes awhile to run
tsne_dt <- data.table(tsne$Y)

#readd the tsne values to master
master$tsne1 <- tsne_dt$V1
master$tsne2 <- tsne_dt$V2
master$tsne3 <- tsne_dt$V3

#only keep tsne values and reddit nums and add top 10 pca values - went from 512 dimensions to 13 dimensions
master1 <- master[,.(tsne1,tsne2,tsne3,reddit)]
master1 <- cbind(master1,pca_dt[,1:10])

#separate back into train and test
train <- master1[!(reddit =='11')]
test <- master1[reddit =='11']

#write out files
fwrite(train, "./FinalProject/volume/data/interim/train.csv")
fwrite(test, "./FinalProject/volume/data/interim/test.csv")
