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

## 
# 튜닝할  것
# 인자
# 범위
# 변수 뭐 넣을 지
# iter

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
      
      cv_pred = predict(cv_xgb,newdata = xgb_cv_te)
      cv_pred = ifelse(cv_pred >= 0.5, 1, 0)
      temp_acc = Accuracy(cv_pred, as.integer(cv_te_label))
      acc = c(acc,temp_acc)
    }
    param_set[i,'ACC'] = mean(acc)
  }
  return(param_set)
}

## ---------- mean --------- ##
# xgboost setting
train_mean <- train_mean_jo_sum_final %>% mutate(Party = as.character(Party))
test_mean <- test_mean_jo_sum

# train_mean_onehot <- dummy.data.frame(train_mean[1:9], dummy.classes = "factor")
# train_mean_onehot <- cbind(train_mean_onehot,train_mean[10:108] %>% mutate_if(is.factor, as.integer)-1,train_mean[109:114])
train_mean_onehot <- dummy.data.frame(train_mean, dummy.classes = "factor")
# test_mean_onehot <- dummy.data.frame(test_mean[1:8], dummy.classes = "factor")
# test_mean_onehot <- cbind(test_mean_onehot, test_mean[9:107] %>% mutate_if(is.factor, as.integer)-1, test_mean[108:113])
test_mean_onehot <- dummy.data.frame(test_mean, dummy.classes = "factor")

train_mean_label <- train_mean_onehot$Party
train_mean_label <- as.numeric(train_mean_label)

xgb_matrix_train_mean <- xgb.DMatrix(as.matrix(select(train_mean_onehot, -Party)), label = train_mean_label)
xgb_matrix_test_mean <- xgb.DMatrix(as.matrix(test_mean_onehot))

# random tuning 
set.seed(1398)
param_set = data.frame(
  max_depth = sample(seq(3,8),20,replace = TRUE),
  min_child_weight = sample(seq(3,8),20,replace = TRUE),
  subsample = runif(20,0.6,1),
  colsample_bytree = runif(20,0.6,1),
  eta = runif(20,0.01,0.3),
  nrounds = sample(c(500,600,700,800),10,replace = TRUE),
  ACC = rep(NA,20),
  objective = "binary:logistic",
  eval_metric = 'error',
  gamma = 0
)

best_param_mean <- xgb_random(iter = 20, param_set = param_set, train_onehot = train_mean_onehot, train_label = train_mean_label)
best_param_mean <- best_param_mean[order(-best_param_mean$ACC),]; best_param_mean[1,]

xg_watchlist_mean <- list(train = xgb_matrix_train_mean)
model_xgb_mean <- xgb.train(params = best_param_mean[1,] %>% as.list(),
                            data = xgb_matrix_train_mean,
                            watchlist = xg_watchlist_mean,
                            verbose = T,
                            print_every_n = 10,
                            nrounds = best_param_mean[1,'nrounds'],
                            early_stopping_rounds = 0.05*best_param_mean[1,'nrounds']
)

predict_xgb_mean <- predict(model_xgb_mean, xgb_matrix_test_mean)
predict_xgb_mean <- ifelse(predict_xgb_mean>=0.5, 1,0)
predict_xgb_mean_class <-  ifelse(predict_xgb_mean==1, "Democrat", "Republican")
submission_mean <- data.frame(test_mean_jo_sum$USER_ID, predict_xgb_mean_class)
colnames(submission_mean) <- c("USER_ID", "Predictions")
submission_mean %>% glimpse()


# submission
fwrite(submission_mean, file = "./submission/submission_mean_xgb.csv", row.names = F) 

## ---------- na --------- ##
# xgboost setting
train_na <- train_na_jo_sum_final %>% mutate(Party = as.character(Party))
test_na <- test_na_jo_sum

train_na_onehot <- dummy.data.frame(train_na, dummy.classes = "factor")
test_na_onehot <- dummy.data.frame(test_na, dummy.classes = "factor")

train_na_label <- train_na_onehot$Party
train_na_label <- as.numeric(train_na_label)

xgb_matrix_train_na <- xgb.DMatrix(as.matrix(select(train_na_onehot, -Party)), label = train_na_label)
xgb_matrix_test_na <- xgb.DMatrix(as.matrix(test_na_onehot))

# random tuning 
set.seed(1398)
param_set = data.frame(
  max_depth = sample(seq(3,8),20,replace = TRUE),
  min_child_weight = sample(seq(3,8),20,replace = TRUE),
  subsample = runif(20,0.6,1),
  colsample_bytree = runif(20,0.6,1),
  eta = runif(20,0.01,0.3),
  nrounds = sample(c(500,600,700,800),10,replace = TRUE),
  ACC = rep(NA,20),
  objective = "binary:logistic",
  eval_metric = 'error',
  gamma = 0
)


best_param_na <- xgb_random(iter = 1, param_set = param_set, train_onehot = train_na_onehot, train_label = train_na_label)
best_param_na <- best_param_na[order(-best_param_na$ACC),]; best_param_na[1,]

xg_watchlist_na <- list(train = xgb_matrix_train_na)
model_xgb_na <- xgb.train(params = best_param_na[1,] %>% as.list(),
                          data = xgb_matrix_train_na,
                          watchlist = xg_watchlist_na,
                          verbose = T,
                          print_every_n = 10,
                          nrounds = best_param_na[1,'nrounds'],
                          early_stopping_rounds = 0.05*best_param_na[1,'nrounds']
)

predict_xgb_na <- predict(model_xgb_na, xgb_matrix_test_na)
predict_xgb_na <- ifelse(predict_xgb_na>=0.5, 1,0)
predict_xgb_na_class <-  ifelse(predict_xgb_na==1, "Democrat", "Republican")
submission_na <- data.frame(test_na_jo_sum$USER_ID, predict_xgb_na_class)
colnames(submission_na) <- c("USER_ID", "Predictions")
submission_na %>% glimpse()


# submission
fwrite(submission_na, file = "./submission/submission_na_xgb.csv", row.names = F) 

## ---------- mca --------- ##
# xgboost setting
train_mca <- train_mca_jo_sum_final %>% mutate(Party = as.character(Party))
test_mca <- test_mca_jo_sum

train_mca_onehot <- dummy.data.frame(train_mca, dummy.classes = "factor")
test_mca_onehot <- dummy.data.frame(test_mca, dummy.classes = "factor")

train_mca_label <- train_mca_onehot$Party
train_mca_label <- as.numeric(train_mca_label)

xgb_matrix_train_mca <- xgb.DMatrix(as.matrix(select(train_mca_onehot, -Party)), label = train_mca_label)
xgb_matrix_test_mca <- xgb.DMatrix(as.matrix(test_mca_onehot))

# random tuning 
set.seed(1398)
param_set = data.frame(
  max_depth = sample(seq(3,8),20,replace = TRUE),
  min_child_weight = sample(seq(3,8),20,replace = TRUE),
  subsample = runif(20,0.6,1),
  colsample_bytree = runif(20,0.6,1),
  eta = runif(20,0.01,0.3),
  nrounds = sample(c(500,600,700,800),10,replace = TRUE),
  ACC = rep(NA,20),
  objective = "binary:logistic",
  eval_metric = 'error',
  gamma = 0
)


best_param_mca <- xgb_random(iter = 1, param_set = param_set, train_onehot = train_mca_onehot, train_label = train_mca_label)
best_param_mca <- best_param_mca[order(-best_param_mca$ACC),]; best_param_mca[1,]

xg_watchlist_mca <- list(train = xgb_matrix_train_mca)
model_xgb_mca <- xgb.train(params = best_param_mca[1,] %>% as.list(),
                           data = xgb_matrix_train_mca,
                           watchlist = xg_watchlist_mca,
                           verbose = T,
                           print_every_n = 10,
                           nrounds = best_param_mca[1,'nrounds'],
                           early_stopping_rounds = 0.05*best_param_mca[1,'nrounds']
)

predict_xgb_mca <- predict(model_xgb_mca, xgb_matrix_test_mca)
predict_xgb_mca <- ifelse(predict_xgb_mca>=0.5, 1,0)
predict_xgb_mca_class <-  ifelse(predict_xgb_mca==1, "Democrat", "Republican")
submission_mca <- data.frame(test_mca_jo_sum$USER_ID, predict_xgb_mca_class)
colnames(submission_mca) <- c("USER_ID", "Predictions")
submission_mca %>% glimpse()


# submission
fwrite(submission_mca, file = "./submission/submission_mca_xgb.csv", row.names = F) 

## ---------- rf --------- ##
# xgboost setting
train_rf <- train_rf_jo_sum_final %>% mutate(Party = as.character(Party))
test_rf <- test_rf_jo_sum

train_rf_onehot <- dummy.data.frame(train_rf, dummy.classes = "factor")
test_rf_onehot <- dummy.data.frame(test_rf, dummy.classes = "factor")

train_rf_label <- train_rf_onehot$Party
train_rf_label <- as.numeric(train_rf_label)

xgb_matrix_train_rf <- xgb.DMatrix(as.matrix(select(train_rf_onehot, -Party)), label = train_rf_label)
xgb_matrix_test_rf <- xgb.DMatrix(as.matrix(test_rf_onehot))

# random tuning 
set.seed(1398)
param_set = data.frame(
  max_depth = sample(seq(3,8),20,replace = TRUE),
  min_child_weight = sample(seq(3,8),20,replace = TRUE),
  subsample = runif(20,0.6,1),
  colsample_bytree = runif(20,0.6,1),
  eta = runif(20,0.01,0.3),
  nrounds = sample(c(500,600,700,800),10,replace = TRUE),
  ACC = rep(NA,20),
  objective = "binary:logistic",
  eval_metric = 'error',
  gamma = 0
)


best_param_rf <- xgb_random(iter = 1, param_set = param_set, train_onehot = train_rf_onehot, train_label = train_rf_label)
best_param_rf <- best_param_rf[order(-best_param_rf$ACC),]; best_param_rf[1,]

xg_watchlist_rf <- list(train = xgb_matrix_train_rf)
model_xgb_rf <- xgb.train(params = best_param_rf[1,] %>% as.list(),
                          data = xgb_matrix_train_rf,
                          watchlist = xg_watchlist_rf,
                          verbose = T,
                          print_every_n = 10,
                          nrounds = best_param_rf[1,'nrounds'],
                          early_stopping_rounds = 0.05*best_param_rf[1,'nrounds']
)

predict_xgb_rf <- predict(model_xgb_rf, xgb_matrix_test_rf)
predict_xgb_rf <- ifelse(predict_xgb_rf>=0.5, 1,0)
predict_xgb_rf_class <-  ifelse(predict_xgb_rf==1, "Democrat", "Republican")
submission_rf <- data.frame(test_rf_jo_sum$USER_ID, predict_xgb_rf_class)
colnames(submission_rf) <- c("USER_ID", "Predictions")
submission_rf %>% glimpse()


# submission
fwrite(submission_rf, file = "./submission/submission_rf_xgb.csv", row.names = F) 

## ---------- nonanswer --------- ##
# xgboost setting with glm lasso
#train_nonanswer_sel <- train_nonanswer_jo_sum_final %>% select(Party,Gender,HouseholdStatus,marriage,re_Q_jealous,Life_Q_drink,edU_Q_mas_doc_degree,ps_Q_Science_Art,Life_Q_readBook,ps_Q_happy_right, ps_Q_Rules, Life_Q_gun, ps_Q_ChangedPersonality, Life_Q_medipray, Life_Q_MacPC, ps_Q_Feminist, re_Q_Dad_householdpower, ps_Q_Overweight, env_Q_p_spank, ps_Q_LifePurpose, careness)

#test_nonanswer_sel <- test_nonanswer_jo_sum %>% select(Gender,HouseholdStatus,marriage,re_Q_jealous,Life_Q_drink,edU_Q_mas_doc_degree,ps_Q_Science_Art,Life_Q_readBook,ps_Q_happy_right, ps_Q_Rules, Life_Q_gun, ps_Q_ChangedPersonality, Life_Q_medipray, Life_Q_MacPC, ps_Q_Feminist, re_Q_Dad_householdpower, ps_Q_Overweight, env_Q_p_spank, ps_Q_LifePurpose, careness)

#train_nonanswer_sel <- train_nonanswer_sel %>% mutate(Party = as.character(Party))

#train_nonanswer_onehot <- dummy.data.frame(train_nonanswer_sel, dummy.classes = "factor")
#test_nonanswer_onehot <- dummy.data.frame(test_nonanswer_sel, dummy.classes = "factor")
 
# train_nonanswer_label <- train_nonanswer_onehot$Party
# train_nonanswer_label <- as.numeric(train_nonanswer_label)

# xgb_matrix_train_nonanswer <- xgb.DMatrix(as.matrix(select(train_nonanswer_onehot, -Party)), label = train_nonanswer_label)
# xgb_matrix_test_nonanswer <- xgb.DMatrix(as.matrix(test_nonanswer_onehot))

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
predict_xgb_nonanswer <- ifelse(predict_xgb_nonanswer>=0.5, 1,0)
predict_xgb_nonanswer_class <-  ifelse(predict_xgb_nonanswer==1, "Democrat", "Republican")
submission_nonanswer <- data.frame(test_nonanswer_jo_sum$USER_ID, predict_xgb_nonanswer_class)
colnames(submission_nonanswer) <- c("USER_ID", "Predictions")
submission_nonanswer %>% glimpse()

# submission
fwrite(submission_nonanswer, file = "./submission/submission_nonanswer_xgb.csv", row.names = F) 

# shap value
library(SHAPforxgboost)
shap_values <- shap.values(xgb_model = model_xgb_nonanswer, X_train = as.matrix(select(train_nonanswer_onehot, -Party)))

shap_long_iris <- shap.prep(xgb_model = model_xgb_nonanswer, X_train = as.matrix(select(train_nonanswer_onehot, -Party)))

shap.plot.summary(shap_long_iris)

shap.plot.summary.wrap2(shap_values$shap_score, as.matrix(select(train_nonanswer_onehot, -Party)))
