# xgboost nonanswer 
# ------ 변수 몽땅 지우기 ------ #
rm(list = ls())
# ------ basic package ------ #
library(tidyverse)
library(plyr)
library(magrittr)
library(data.table)
library(gridExtra)

# ------ choose working directory ------ #
setwd("C:/Users/user/Desktop/찬영/VISION/학회/P-SAT/주제분석/20년도 2학기 26기/주제분석 2-3주차/modeling data")
getwd()

# ------ load data ------ #
load("C:/Users/user/Desktop/찬영/VISION/학회/P-SAT/주제분석/20년도 2학기 26기/주제분석 2-3주차/load_data_WEEK3.RData")

# ------ XGboost 가즈아!!! ------- ##
library(dummies)
library(xgboost)
library(caret)
library(MLmetrics)
library(ROCR)
# ------ load data ------ #
load("C:/Users/user/Desktop/찬영/VISION/학회/P-SAT/주제분석/20년도 2학기 26기/주제분석 2-3주차/load_data_WEEK3.RData")
rm(list = setdiff(ls(), c("train_nonanswer_jo_sum_final", "test_nonanswer_jo_sum")))
# ------ random tuning ------- #
xgb_random <- function(iter, param_set, train_onehot, train_label){
  for (i in 1:iter) {
    params = param_set[i,] %>%
      select(-c(ACC,nrounds)) %>%
      as.list()
    
    set.seed(1398)
    cv = createFolds(train_label, k = 5)
    acc = NULL
    
    for (j in 1:5) {
      valid_idx = cv[[j]]
      cv_tr_dummy = train_onehot[-valid_idx,]
      cv_te_dummy = train_onehot[valid_idx,]
      cv_tr_label = cv_tr_dummy$Party
      cv_te_label = cv_te_dummy$Party
      xgb_cv_tr = xgb.DMatrix(as.matrix(select(cv_tr_dummy,-Party)),label = cv_tr_label)
      xgb_cv_te = xgb.DMatrix(as.matrix(select(cv_te_dummy,-Party)),label = cv_te_label)
      watchlist <- list(train = xgb_cv_tr, test = xgb_cv_te)
      
      
      set.seed(1398)
      cv_xgb = xgb.train(
        params = params,
        data = xgb_cv_tr,
        nrounds = param_set[i,'nrounds'],
        early_stopping_rounds = 0.05*param_set[i,'nrounds'],
        watchlist = watchlist,
        verbose = T)

      cv_pred = predict(cv_xgb, newdata = xgb_cv_te)
      
      pr_full <- prediction(cv_pred, cv_te_label)
      prf_full <- performance(pr_full, measure = "tpr", x.measure = "fpr")
      
      optid_full <- (1:length(prf_full@y.values[[1]][-1]))[((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-11])^2)==min((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-1])^2)]

      optcut_full <- prf_full@alpha.values[[1]][-1][optid_full]
      
      cv_pred = ifelse(cv_pred >= optcut_full, 1, 0)
      temp_acc = Accuracy(cv_pred, as.integer(cv_te_label))
      acc = c(acc,temp_acc)
    }
    param_set[i,'ACC'] = mean(acc)
    param_set[i,'cutoff'] = optcut_full
  }
  return(param_set)
}

# xgboost setting
# 중요 변수 선택! -> 변수 선택 안하는 게 훨씬 성능 좋음
# xgboost setting
train_nonanswer <- train_nonanswer_jo_sum_final %>% mutate(Party = as.character(Party))
test_nonanswer <- test_nonanswer_jo_sum

train_nonanswer_onehot <- dummy.data.frame(train_nonanswer, dummy.classes = "factor")
test_nonanswer_onehot <- dummy.data.frame(test_nonanswer, dummy.classes = "factor")

train_nonanswer_label <- train_nonanswer_onehot$Party
train_nonanswer_label <- as.numeric(train_nonanswer_label)

xgb_matrix_train_nonanswer <- xgb.DMatrix(as.matrix(select(train_nonanswer_onehot, -Party)), label = train_nonanswer_label)
xgb_matrix_test_nonanswer <- xgb.DMatrix(as.matrix(test_nonanswer_onehot))

# random tuning 
set.seed(1398)
param_set = data.frame(
  max_depth = sample(seq(3,8),20,replace = TRUE),
  min_child_weight = sample(seq(3,8),20,replace = TRUE),
  subsample = runif(20,0.6,1),
  colsample_bytree = runif(20,0.6,1),
  eta = runif(20,0.01,0.3),
  nrounds = sample(c(200,250,300,450),10,replace = TRUE),
  ACC = rep(NA,20),
  cutoff = rep(NA,20),
  objective = "binary:logistic",
  eval_metric = 'error',
  gamma = 0
)


best_param_nonanswer <- xgb_random(iter = 20, param_set = param_set, train_onehot = train_nonanswer_onehot, train_label = train_nonanswer_label)
best_param_nonanswer <- best_param_nonanswer[order(-best_param_nonanswer$ACC),]; best_param_nonanswer[1,]

xg_watchlist_nonanswer <- list(train = xgb_matrix_train_nonanswer)
model_xgb_nonanswer <- xgb.train(params = best_param_nonanswer[1,] %>% as.list(),
                                 data = xgb_matrix_train_nonanswer,
                                 watchlist = xg_watchlist_nonanswer,
                                 verbose = T,
                                 print_every_n = 10,
                                 nrounds = best_param_nonanswer[1,'nrounds'],
                                 early_stopping_rounds = 0.05*best_param_nonanswer[1,'nrounds']
)

predict_xgb_nonanswer <- predict(model_xgb_nonanswer, xgb_matrix_test_nonanswer)
predict_xgb_nonanswer <- ifelse(predict_xgb_nonanswer>=best_param_nonanswer[,"cutoff"][1], 1,0)
predict_xgb_nonanswer_class <-  ifelse(predict_xgb_nonanswer==1, "Democrat", "Republican")
submission_nonanswer <- data.frame(test_nonanswer_jo_sum$USER_ID, predict_xgb_nonanswer_class)
colnames(submission_nonanswer) <- c("USER_ID", "Predictions")
submission_nonanswer %>% glimpse()

# # submission
# fwrite(submission_nonanswer, file = "./submission/submission_nonanswer_xgb_5.csv", row.names = F)

# ------ grid search ------- #
# xgb_grid
best_param_nonanswer <- best_param_nonanswer_0645
# grid 하고 싶은거 정하고,
max_depth = seq(best_param_nonanswer[1,]$max_depth-2,best_param_nonanswer[1,]$max_depth+2,1)
min_child_weight = seq(best_param_nonanswer[1,]$min_child_weight-2,best_param_nonanswer[1,]$min_child_weight+2,1)

grid_1 = data.frame(
  max_depth = rep(max_depth,2),
  min_child_weight = rep(min_child_weight,each = 2),
  ACC = rep(NA,10),
  cutoff = rep(NA,10)
)

xgb_grid <- function(iter, param_set, train_onehot, train_label){
  for (i in 1:iter) {
    param_1 = list(
      max_depth = grid_1[i,'max_depth'],
      min_child_weight = grid_1[i,'min_child_weight'],
      subsample = 1,
      colsample_bytree = 1,
      eta = 0.1
    )
    
    set.seed(1398)
    cv = createFolds(train_label, k = 5)
    acc = NULL
    
    for (j in 1:5) {
      valid_idx = cv[[j]]
      cv_tr_dummy = train_onehot[-valid_idx,]
      cv_te_dummy = train_onehot[valid_idx,]
      cv_tr_label = cv_tr_dummy$Party
      cv_te_label = cv_te_dummy$Party
      xgb_cv_tr = xgb.DMatrix(as.matrix(select(cv_tr_dummy,-Party)),label = cv_tr_label)
      xgb_cv_te = xgb.DMatrix(as.matrix(select(cv_te_dummy,-Party)),label = cv_te_label)
      watchlist <- list(train = xgb_cv_tr, test = xgb_cv_te)
      
      
      set.seed(1398)
      cv_xgb = xgb.train(
        params = param_1,
        data = xgb_cv_tr,
        nrounds = param_set$nrounds,
        early_stopping_rounds = 0.05*param_set$nrounds,
        watchlist = watchlist,
        verbose = T)
      
      cv_pred = predict(cv_xgb, newdata = xgb_cv_te)
      
      pr_full <- prediction(cv_pred, cv_te_label)
      prf_full <- performance(pr_full, measure = "tpr", x.measure = "fpr")
      
      optid_full <- (1:length(prf_full@y.values[[1]][-1]))[((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-11])^2)==min((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-1])^2)]
      
      optcut_full <- prf_full@alpha.values[[1]][-1][optid_full]
      
      cv_pred = ifelse(cv_pred >= optcut_full, 1, 0)
      temp_acc = Accuracy(cv_pred, as.integer(cv_te_label))
      acc = c(acc,temp_acc)
    }
    param_set[i,'ACC'] = mean(acc)
    param_set[i,'cutoff'] = optcut_full
  }
  return(param_set)
}
best_grid <- xgb_grid(10, param_set = best_param_nonanswer[1,], train_onehot = train_nonanswer_onehot, train_label = train_nonanswer_label)
best_grid <- best_grid[order(-best_param_nonanswer$ACC),]; best_grid[1,]

xg_watchlist_nonanswer <- list(train = xgb_matrix_train_nonanswer)
model_xgb_nonanswer <- xgb.train(params = best_grid[1,] %>% as.list(),
                                 data = xgb_matrix_train_nonanswer,
                                 watchlist = xg_watchlist_nonanswer,
                                 verbose = T,
                                 print_every_n = 10,
                                 nrounds = best_grid[1,'nrounds'],
                                 early_stopping_rounds = 0.05*best_grid[1,'nrounds']
)

predict_xgb_nonanswer <- predict(model_xgb_nonanswer, xgb_matrix_test_nonanswer)
predict_xgb_nonanswer <- ifelse(predict_xgb_nonanswer>=best_grid[,"cutoff"][1], 1,0)
predict_xgb_nonanswer_class <-  ifelse(predict_xgb_nonanswer==1, "Democrat", "Republican")
submission_nonanswer <- data.frame(test_nonanswer_jo_sum$USER_ID, predict_xgb_nonanswer_class)
colnames(submission_nonanswer) <- c("USER_ID", "Predictions")
submission_nonanswer %>% glimpse()

# submission
fwrite(submission_nonanswer, file = "./submission/submission_nonanswer_xgb_7.csv", row.names = F)

