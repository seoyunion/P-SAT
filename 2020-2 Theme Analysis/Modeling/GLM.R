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

# ------ train data load ------ #
# 취합된 데이터 불러오고, work space 제작하자!
train_mean_jo_sum_final <- fread(file = "train_mean_jo_sum_final.csv", header = T, stringsAsFactors = F, data.table = F)
train_rf_jo_sum_final <- fread(file = "train_rf_jo_sum_final.csv", header = T, stringsAsFactors = F, data.table = F)

train_nonanswer_jo_sum_final <- fread(file = "train_nonanswer_jo_sum_final.csv", header = T, stringsAsFactors = F, data.table = F)

train_mca_jo_sum_final <- fread(file = "train_mca_jo_sum_final.csv", header = T, stringsAsFactors = F, data.table = F)

train_na_jo_sum_final <- fread(file = "train_na_jo_sum_final.csv", header = T, stringsAsFactors = F, data.table = F)

# ------ test data load ------ #
test_mean_jo_sum <- fread(file =  "test_mean_jo_sum.csv", header = T, stringsAsFactors = F, data.table = F)
test_rf_jo_sum <- fread(file =  "test_rf_jo_sum.csv", header = T, stringsAsFactors = F, data.table = F)
test_nonanswer_jo_sum <- fread(file =  "test_nonanswer_jo_sum.csv", header = T, stringsAsFactors = F, data.table = F)
test_mca_jo_sum <- fread(file =  "test_mca_jo_sum.csv", header = T, stringsAsFactors = F, data.table = F)
test_na_jo_sum <- fread(file =  "test_na_jo_sum.csv", header = T, stringsAsFactors = F, data.table = F)

# ------ mean_jo_sum_final data 형식 맞추기 ------ #
# train set
# 질문 변수 factor
train_mean_jo_sum_final[10:108] %<>% mutate_if(is.integer, as.factor)
# 인적 변수 factor (소득수준 및 교육수준 제외)
train_mean_jo_sum_final[1:9] %<>% mutate_if(is.character, as.factor)
train_mean_jo_sum_final$Party <-  ifelse(train_mean_jo_sum_final$Party=="Democrat", 1, 0)  %>% as.factor()
train_mean_jo_sum_final %>% glimpse()

# test set
# 질문 변수 factor
test_mean_jo_sum[9:107] %<>% mutate_if(is.integer, as.factor)
# 인적 변수 factor (소득수준 및 교육수준 제외)
test_mean_jo_sum[1:8] %<>% mutate_if(is.character, as.factor)
test_mean_jo_sum %>% glimpse()

# ------ train_rf_jo_sum_final data 형식 맞추기 ------ #
# train set
# 소득수준, 교육수준 파생변수 외에 모두 범주형 변수로!
train_rf_jo_sum_final %<>% mutate_if(is.character, as.factor)
# ------ train_rf_jo_sum_final data 형식 맞추기 ------ ## glm 돌리기 위해 숫자 형태로 변경
train_rf_jo_sum_final$Party <-  ifelse(train_rf_jo_sum_final$Party=="Democrat", 1, 0)  %>% as.factor()
train_rf_jo_sum_final %>% glimpse()

# test set
test_rf_jo_sum %<>% mutate_if(is.character, as.factor)
test_rf_jo_sum %>% glimpse()

# ------ train_mca_jo_sum_final data 형식 맞추기 ------ #
# train set
# 소득수준, 교육수준 파생변수 외에 모두 범주형 변수로!
train_mca_jo_sum_final %<>% mutate_if(is.character, as.factor)
# ------ train_rf_jo_sum_final data 형식 맞추기 ------ ## glm 돌리기 위해 숫자 형태로 변경
train_mca_jo_sum_final$Party <-  ifelse(train_mca_jo_sum_final$Party=="Democrat", 1, 0)  %>% as.factor()
train_mca_jo_sum_final %>% glimpse()

# test set
test_mca_jo_sum %<>% mutate_if(is.character, as.factor)
test_mca_jo_sum %>% glimpse()
# ------ train_nonanswer_jo_sum_final data 형식 맞추기 ------ #
# train set
# 소득수준, 교육수준 파생변수 외에 모두 범주형 변수로!
train_nonanswer_jo_sum_final %<>% mutate_if(is.character, as.factor)
# ------ train_rf_jo_sum_final data 형식 맞추기 ------ ## glm 돌리기 위해 숫자 형태로 변경
train_nonanswer_jo_sum_final$Party <-  ifelse(train_nonanswer_jo_sum_final$Party=="Democrat", 1, 0)  %>% as.factor()
train_nonanswer_jo_sum_final %>% glimpse()

# test set
test_nonanswer_jo_sum %<>% mutate_if(is.character, as.factor)
test_nonanswer_jo_sum %>% glimpse()

# ------ train_na_jo_sum_final data 형식 맞추기 ------ #
# train set
# 소득수준, 교육수준 파생변수 외에 모두 범주형 변수로!
train_na_jo_sum_final %<>% mutate_if(is.character, as.factor)
# ------ train_rf_jo_sum_final data 형식 맞추기 ------ ## glm 돌리기 위해 숫자 형태로 변경
train_na_jo_sum_final$Party <-  ifelse(train_na_jo_sum_final$Party=="Democrat", 1, 0)  %>% as.factor()
train_na_jo_sum_final %>% glimpse()

# test set
test_na_jo_sum %<>% mutate_if(is.character, as.factor)
test_na_jo_sum %>% glimpse()

modeling_list <- c("test_mca_jo_sum", "test_mean_jo_sum", "test_na_jo_sum", "test_nonanswer_jo_sum", "test_rf_jo_sum", "train_mca_jo_sum_final", "train_mean_jo_sum_final", "train_na_jo_sum_final", "train_nonanswer_jo_sum_final", "train_rf_jo_sum_final") 


#load("C:/Users/user/Desktop/찬영/VISION/학회/P-SAT/주제분석/20년도 2학기 26기/주제분석 2-3주차/load_data_WEEK3.RData")

## -------- Modeling ------------------ ##
library(glmnet) # logistic package
library(lmtest)
library(caret)
library(MLmetrics)
library(ROCR)
library(dummies)
# 변수선택법은 너무 많은 computing power 소요
# Ridge는 변수선택 할 수 없고 돌려보니 lasso에 비해 정확도 떨어짐
# glm이다보니 선형독립 만족 안하면 NA로 뜸 Lasso penalty를 활용하여 다중공선성을 제거하고 변수선택 해보자!

## ------------- mean_jo_sum_data ------------ ##
# Split the data into training and test set
set.seed(1398)
index <- train_mean_jo_sum_final$Party %>% createDataPartition(p = .8, list = F)
train_mean <- train_mean_jo_sum_final[index,]
valid_mean <- train_mean_jo_sum_final[-index,]
# data matrix 형태로 만들어주기!
train_mean_x <- model.matrix(Party~., train_mean)[,-1]
train_mean_y <- train_mean$Party

# 로지스틱 with lasso적합 alpha = 1
# 최적의 람다 값 탐색!
# Find the best lambda using cross-validation
set.seed(1398) 
cv.lasso <- cv.glmnet(train_mean_x, train_mean_y, alpha = 1, family = "binomial", type.measure = 'class', standardized = T)
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar = "lambda",label = F)

opt_lam_mean <- cv.lasso$lambda.min # 오분류율 가장 낮게 하는 값
cv.lasso$lambda.1se # 표준오차가 가장 정규화되는 모델이 되게하는 람다 값

# 각 람다값에 따른 선택된 변수들
coef(cv.lasso, cv.lasso$lambda.min)
coef(cv.lasso, cv.lasso$lambda.1se)

# Final model with lambda.min
mean_lasso_min <- glmnet(train_mean_x, train_mean_y, alpha = 1, family = "binomial", lambda = opt_lam_mean)

# Make prediction on valid data
valid_mean_x <- model.matrix(Party ~., valid_mean)[,-1]
pred_mean_min <- mean_lasso_min %>% predict(newx = valid_mean_x, type = 'response')

# 최적의 cut-off값 탐색!
pr_mean_min <- prediction(pred_mean_min, valid_mean$Party)
prf_mean_min <- performance(pr_mean_min, measure = "tpr", x.measure = "fpr")

optid_mean_min <- (1:length(prf_mean_min@y.values[[1]][-1]))[((prf_mean_min@x.values[[1]][-1])^2 + (1-prf_mean_min@y.values[[1]][-11])^2)==min((prf_mean_min@x.values[[1]][-1])^2 + (1-prf_mean_min@y.values[[1]][-1])^2)]

# (0,1)에 가장 가까운 값 탐색!
optcut_mean_min_x <- prf_mean_min@x.values[[1]][-1][optid_mean_min]
optcut_mean_min_y <- prf_mean_min@y.values[[1]][-1][optid_mean_min]
optcut_mean_min <- prf_mean_min@alpha.values[[1]][-1][optid_mean_min]

ggplot() +
  geom_point(aes(x = optcut_mean_min_x, 
                 y = optcut_mean_min_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_mean_min_x, 
                y = optcut_mean_min_y, label = prf_mean_min@alpha.values[[1]][-1][optid_mean_min] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_mean_min@x.values[[1]],
                y = prf_mean_min@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_min_class <- ifelse(pred_mean_min >= optcut_mean_min, 1, 0)

# Model accuracy
observed_class <- valid_mean$Party
acc_mean_min <- mean(pred_min_class == observed_class)

# Fit the model 단순 전체 적합이랑 비교!
full_model <- glm(Party ~., data = train_mean, family = binomial)
# Make predictions
pred_full <- full_model %>% predict(valid_mean, type = "response")

pr_full <- prediction(pred_full, valid_mean$Party)
prf_full <- performance(pr_full, measure = "tpr", x.measure = "fpr")

optid_full <- (1:length(prf_full@y.values[[1]][-1]))[((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-11])^2)==min((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-1])^2)]

optcut_full_x <- prf_full@x.values[[1]][-1][optid_full]
optcut_full_y <- prf_full@y.values[[1]][-1][optid_full]
optcut_full <- prf_full@alpha.values[[1]][-1][optid_full]

ggplot() +
  geom_point(aes(x = optcut_full_x, 
                 y = optcut_full_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_full_x, 
                y = optcut_full_y, label = prf_full@alpha.values[[1]][-1][optid_full] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_full@x.values[[1]],
                y = prf_full@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_full_class <- ifelse(pred_full >= optcut_full, 1, 0)

# Model accuracy
observed_class <- valid_mean$Party
acc_full <- mean(pred_full_class == observed_class)

# lasso 적용했을 때 더 높은 정확도 보인다! acc_mean_min

# Make prediction on test data
# test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
full_model <- glm(Party ~., data = train_mean_jo_sum_final, family = binomial(link = "logit"))
pred_mean_min_test <- full_model %>% predict(test_mean_jo_sum, type = 'response')
pred_min_class_test <- ifelse(pred_mean_min_test >= optcut_full, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_mean <- data.frame(test_mean_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_mean) <- c("USER_ID", "Predictions")
submission_mean %>% glimpse()

# submission
fwrite(submission_mean, file = "./submission/submission_mean_full.csv", row.names = F) # 



# 전체 적합해서 진행
# data matrix 형태로 만들어주기!
train_mean_x <- model.matrix(Party~., train_mean_jo_sum_final)[,-1]
train_mean_y <- train_mean_jo_sum_final$Party

mean_lasso_min <- glmnet(train_mean_x, train_mean_y, alpha = 1, family = "binomial", lambda = opt_lam_mean)

# test pred
test_mean_x <- model.matrix(~.,test_mean_jo_sum)[,-1]
pred_mean_min_test <- mean_lasso_min %>% predict(newx = test_mean_x, type = 'response')
pred_min_class_test <- ifelse(pred_mean_min_test >= optcut_mean_min, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_mean <- data.frame(test_mean_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_mean) <- c("USER_ID", "Predictions")
submission_mean %>% glimpse()

# submission
fwrite(submission_mean, file = "./submission/submission_mean_final.csv", row.names = F) #

rm(list = setdiff(ls(), c("test_mca_jo_sum", "test_mean_jo_sum", "test_na_jo_sum", "test_nonanswer_jo_sum", "test_rf_jo_sum", "train_mca_jo_sum_final", "train_mean_jo_sum_final", "train_na_jo_sum_final", "train_nonanswer_jo_sum_final", "train_rf_jo_sum_final")))

## ------------- rf_jo_sum_data ------------ ##
# Split the data into training and test set
set.seed(1398)
index <- train_rf_jo_sum_final$Party %>% createDataPartition(p = .8, list = F)
train_rf <- train_rf_jo_sum_final[index,]
valid_rf <- train_rf_jo_sum_final[-index, ]
# data matrix 형태로 만들어주기!
train_rf_x <- model.matrix(Party~., train_rf)[,-1]
train_rf_y <- train_rf$Party

# 로지스틱 with lasso적합 alpha = 1
# 최적의 람다 값 탐색!
# Find the best lambda using cross-validation
set.seed(1398) 
cv.lasso <- cv.glmnet(train_rf_x, train_rf_y, alpha = 1, family = "binomial", type.measure = 'class', standardized = T)
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar = "lambda",label = F)

opt_lam_rf <- cv.lasso$lambda.min # 오분류율 가장 낮게 하는 값
cv.lasso$lambda.1se # 표준오차가 가장 정규화되는 모델이 되게하는 람다 값

# 각 람다값에 따른 선택된 변수들
coef(cv.lasso, cv.lasso$lambda.min)
coef(cv.lasso, cv.lasso$lambda.1se)

# Final model with lambda.min
rf_lasso_min <- glmnet(train_rf_x, train_rf_y, alpha = 1, family = "binomial", lambda = opt_lam_rf)

# Make prediction on valid data
valid_rf_x <- model.matrix(Party ~., valid_rf)[,-1]
pred_rf_min <- rf_lasso_min %>% predict(newx = valid_rf_x, type = 'response')

# 최적의 cut-off값 탐색!
pr_rf_min <- prediction(pred_rf_min, valid_rf$Party)
prf_rf_min <- performance(pr_rf_min, measure = "tpr", x.measure = "fpr")

optid_rf_min <- (1:length(prf_rf_min@y.values[[1]][-1]))[((prf_rf_min@x.values[[1]][-1])^2 + (1-prf_rf_min@y.values[[1]][-11])^2)==min((prf_rf_min@x.values[[1]][-1])^2 + (1-prf_rf_min@y.values[[1]][-1])^2)]

# (0,1)에 가장 가까운 값 탐색!
optcut_rf_min_x <- prf_rf_min@x.values[[1]][-1][optid_rf_min]
optcut_rf_min_y <- prf_rf_min@y.values[[1]][-1][optid_rf_min]
optcut_rf_min <- prf_rf_min@alpha.values[[1]][-1][optid_rf_min]

ggplot() +
  geom_point(aes(x = optcut_rf_min_x, 
                 y = optcut_rf_min_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_rf_min_x, 
                y = optcut_rf_min_y, label = prf_rf_min@alpha.values[[1]][-1][optid_rf_min] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_rf_min@x.values[[1]],
                y = prf_rf_min@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_min_class <- ifelse(pred_rf_min >= optcut_rf_min, 1, 0)

# Model accuracy
observed_class <- valid_rf$Party
acc_rf_min <- mean(pred_min_class == observed_class)

# Fit the model 단순 전체 적합이랑 비교!
full_model <- glm(Party ~., data = train_rf, family = binomial)
# Make predictions
pred_full <- full_model %>% predict(valid_rf, type = "response")

pr_full <- prediction(pred_full, valid_rf$Party)
prf_full <- performance(pr_full, measure = "tpr", x.measure = "fpr")

optid_full <- (1:length(prf_full@y.values[[1]][-1]))[((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-11])^2)==min((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-1])^2)]

optcut_full_x <- prf_full@x.values[[1]][-1][optid_full]
optcut_full_y <- prf_full@y.values[[1]][-1][optid_full]
optcut_full <- prf_full@alpha.values[[1]][-1][optid_full]

ggplot() +
  geom_point(aes(x = optcut_full_x, 
                 y = optcut_full_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_full_x, 
                y = optcut_full_y, label = prf_full@alpha.values[[1]][-1][optid_full] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_full@x.values[[1]],
                y = prf_full@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_full_class <- ifelse(pred_full >= optcut_full, 1, 0)

# Model accuracy
observed_class <- valid_rf$Party
acc_full <- mean(pred_full_class == observed_class)

# lasso 적용했을 때 더 높은 정확도 보인다! acc_rf_min

# Make prediction on test data
# test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
full_model <- glm(Party ~., data = train_rf_jo_sum_final, family = binomial(link = "logit"))
pred_rf_min_test <- full_model %>% predict(test_rf_jo_sum, type = 'response')
pred_min_class_test <- ifelse(pred_rf_min_test >= optcut_full, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_rf <- data.frame(test_rf_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_rf) <- c("USER_ID", "Predictions")
submission_rf %>% glimpse()

# submission
fwrite(submission_rf, file = "./submission/submission_rf_full.csv", row.names = F) # 

# 전체 적합해서 진행
# data matrix 형태로 만들어주기!
train_mean_x <- model.matrix(Party~., train_rf_jo_sum_final)[,-1]
train_mean_y <- train_rf_jo_sum_final$Party

rf_lasso_min <- glmnet(train_mean_x, train_mean_y, alpha = 1, family = "binomial", lambda = opt_lam_rf)

# Make prediction on test data
test_rf_x <- model.matrix(~.,test_rf_jo_sum)[,-1]
pred_rf_min_test <- rf_lasso_min %>% predict(newx = test_rf_x, type = 'response')
pred_min_class_test <- ifelse(pred_rf_min_test >= optcut_rf_min, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_rf <- data.frame(test_rf_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_rf) <- c("USER_ID", "Predictions")
submission_rf %>% glimpse()

# submission
fwrite(submission_rf, file = "./submission/submission_rf_final.csv", row.names = F) #

rm(list = setdiff(ls(), c("test_mca_jo_sum", "test_mean_jo_sum", "test_na_jo_sum", "test_nonanswer_jo_sum", "test_rf_jo_sum", "train_mca_jo_sum_final", "train_mean_jo_sum_final", "train_na_jo_sum_final", "train_nonanswer_jo_sum_final", "train_rf_jo_sum_final")))

## ------------- mca_jo_sum_data ------------ ##
# Split the data into training and test set
set.seed(1398)
index <- train_mca_jo_sum_final$Party %>% createDataPartition(p = .8, list = F)
train_mca <- train_mca_jo_sum_final[index,]
valid_mca <- train_mca_jo_sum_final[-index, ]
# data matrix 형태로 만들어주기!
train_mca_x <- model.matrix(Party~., train_mca)[,-1]
train_mca_y <- train_mca$Party

# 로지스틱 with lasso적합 alpha = 1
# 최적의 람다 값 탐색!
# Find the best lambda using cross-validation
set.seed(1398) 
cv.lasso <- cv.glmnet(train_mca_x, train_mca_y, alpha = 1, family = "binomial", type.measure = 'class', standardized = T)
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar = "lambda",label = F)

opt_lam_mca <- cv.lasso$lambda.min # 오분류율 가장 낮게 하는 값
cv.lasso$lambda.1se # 표준오차가 가장 정규화되는 모델이 되게하는 람다 값

# 각 람다값에 따른 선택된 변수들
coef(cv.lasso, cv.lasso$lambda.min)
coef(cv.lasso, cv.lasso$lambda.1se)

# Final model with lambda.min
mca_lasso_min <- glmnet(train_mca_x, train_mca_y, alpha = 1, family = "binomial", lambda = opt_lam_mca)

# Make prediction on valid data
valid_mca_x <- model.matrix(Party ~., valid_mca)[,-1]
pred_mca_min <- mca_lasso_min %>% predict(newx = valid_mca_x, type = 'response')

# 최적의 cut-off값 탐색!
pr_mca_min <- prediction(pred_mca_min, valid_mca$Party)
prf_mca_min <- performance(pr_mca_min, measure = "tpr", x.measure = "fpr")

optid_mca_min <- (1:length(prf_mca_min@y.values[[1]][-1]))[((prf_mca_min@x.values[[1]][-1])^2 + (1-prf_mca_min@y.values[[1]][-11])^2)==min((prf_mca_min@x.values[[1]][-1])^2 + (1-prf_mca_min@y.values[[1]][-1])^2)]

# (0,1)에 가장 가까운 값 탐색!
optcut_mca_min_x <- prf_mca_min@x.values[[1]][-1][optid_mca_min]
optcut_mca_min_y <- prf_mca_min@y.values[[1]][-1][optid_mca_min]
optcut_mca_min <- prf_mca_min@alpha.values[[1]][-1][optid_mca_min]

ggplot() +
  geom_point(aes(x = optcut_mca_min_x, 
                 y = optcut_mca_min_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_mca_min_x, 
                y = optcut_mca_min_y, label = prf_mca_min@alpha.values[[1]][-1][optid_mca_min] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_mca_min@x.values[[1]],
                y = prf_mca_min@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_min_class <- ifelse(pred_mca_min >= optcut_mca_min, 1, 0)

# Model accuracy
observed_class <- valid_mca$Party
acc_mca_min <- mean(pred_min_class == observed_class)

# Fit the model 단순 전체 적합이랑 비교!
full_model <- glm(Party ~., data = train_mca, family = binomial)
# Make predictions
pred_full <- full_model %>% predict(valid_mca, type = "response")

pr_full <- prediction(pred_full, valid_mca$Party)
prf_full <- performance(pr_full, measure = "tpr", x.measure = "fpr")

optid_full <- (1:length(prf_full@y.values[[1]][-1]))[((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-11])^2)==min((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-1])^2)]

optcut_full_x <- prf_full@x.values[[1]][-1][optid_full]
optcut_full_y <- prf_full@y.values[[1]][-1][optid_full]
optcut_full <- prf_full@alpha.values[[1]][-1][optid_full]

ggplot() +
  geom_point(aes(x = optcut_full_x, 
                 y = optcut_full_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_full_x, 
                y = optcut_full_y, label = prf_full@alpha.values[[1]][-1][optid_full] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_full@x.values[[1]],
                y = prf_full@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_full_class <- ifelse(pred_full >= optcut_full, 1, 0)

# Model accuracy
observed_class <- valid_mca$Party
acc_full <- mean(pred_full_class == observed_class)

# 그냥 다 적합했을 때 오히려 더 높은 정확도 보인다! acc_full

# 전체 적합해서 진행
# data matrix 형태로 만들어주기!
train_mca_x <- model.matrix(Party~., train_mca_jo_sum_final)[,-1]
train_mca_y <- train_mca_jo_sum_final$Party

mca_lasso_min <- glmnet(train_mca_x, train_mca_y, alpha = 1, family = "binomial", lambda = opt_lam_mca)

# test pred
test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
pred_mca_min_test <- mca_lasso_min %>% predict(newx = test_mca_x, type = 'response')
pred_min_class_test <- ifelse(pred_mca_min_test >= optcut_mca_min, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_mca <- data.frame(test_mca_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_mca) <- c("USER_ID", "Predictions")
submission_mca %>% glimpse()

# submission
fwrite(submission_mca, file = "./submission/submission_mca_final.csv", row.names = F) #


# Make prediction on test data
# test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
full_model <- glm(Party ~., data = train_mca_jo_sum_final, family = binomial(link = "logit"))
pred_mca_min_test <- full_model %>% predict(test_mca_jo_sum, type = 'response')
pred_min_class_test <- ifelse(pred_mca_min_test >= optcut_full, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_mca <- data.frame(test_mca_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_mca) <- c("USER_ID", "Predictions")
submission_mca %>% glimpse()

# submission
fwrite(submission_mca, file = "./submission/submission_mca_full.csv", row.names = F) # 

# 전체 적합해서 진행
# # data matrix 형태로 만들어주기!
# train_mean_x <- model.matrix(Party~., train_mca_jo_sum_final)[,-1]
# train_mean_y <- train_mca_jo_sum_final$Party
# 
# mca_lasso_min <- glmnet(train_mean_x, train_mean_y, alpha = 1, family = "binomial", lambda = opt_lam_mca)
# 
# # Make prediction on test data
# test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
# pred_mca_min_test <- mca_lasso_min %>% predict(newx = test_mca_x, type = 'response')
# pred_min_class_test <- ifelse(pred_mca_min_test >= optcut_rf_min, 1, 0)
# pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
# submission_mca <- data.frame(test_mca_jo_sum$USER_ID, pred_min_class_test)
# colnames(submission_mca) <- c("USER_ID", "Predictions")
# submission_mca %>% glimpse()
# 
# # submission
# fwrite(submission_mca, file = "./submission/submission_mca_final.csv", row.names = F) #


# Fit the model tuning가즈아!!
# full_model_tune <- glm(Party ~-USER_ID+HouseholdStatus+age+edU_Q_mas_doc_degree+Life_Q_readBook+ps_Q_PowerOfPositive+ps_Q_ResultOfCircumstance+ps_Q_ChangedPersonality+Life_Q_medipray+ps_Q_Feminist+edu_Q_parents_college+re_Q_extendedfamilly+re_Q_Dad_householdpower+env_Q_p_spank, data = train_mca, family = binomial(link = "logit"))
# 
# summary(full_model_tune)
# 
# # Make predictions
# pred_full_tune <- full_model_tune %>% predict(valid_mca, type = "response")
# 
# pr_full_tune <- prediction(pred_full_tune, valid_mca$Party)
# prf_full_tune <- performance(pr_full_tune, measure = "tpr", x.measure = "fpr")
# 
# optid_full_tune <- (1:length(prf_full_tune@y.values[[1]][-1]))[((prf_full_tune@x.values[[1]][-1])^2 + (1-prf_full_tune@y.values[[1]][-11])^2)==min((prf_full_tune@x.values[[1]][-1])^2 + (1-prf_full_tune@y.values[[1]][-1])^2)]
# 
# optcut_full_x_tune <- prf_full_tune@x.values[[1]][-1][optid_full_tune]
# optcut_full_y_tune <- prf_full_tune@y.values[[1]][-1][optid_full_tune]
# optcut_full_tune <- prf_full_tune@alpha.values[[1]][-1][optid_full_tune]
# 
# ggplot() +
#   geom_point(aes(x = optcut_full_x_tune, 
#                  y = optcut_full_y_tune),size = 3,  color = "red") +
#   geom_text(aes(x = optcut_full_x_tune, 
#                 y = optcut_full_y_tune, label = prf_full_tune@alpha.values[[1]][-1][optid_full_tune] %>% round(3))
#             ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
#   geom_line(aes(x = prf_full_tune@x.values[[1]],
#                 y = prf_full_tune@y.values[[1]]), size = 1) +
#   labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
# pred_full_class_tune <- ifelse(pred_full_tune >= optcut_full_tune, 1, 0)

# Model accuracy
# observed_class <- valid_mca$Party
# acc_full_tune <- mean(pred_full_class_tune == observed_class)

# Make prediction on test data
# test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
# pred_mca_min_test_tune <- full_model_tune %>% predict(test_mca_jo_sum, type = 'response')
# pred_min_class_test_tune <- ifelse(pred_mca_min_test_tune >= optcut_full_tune, 1, 0)
# pred_min_class_test_tune <-  ifelse(pred_mca_min_test_tune==1, "Democrat", "Republican")
# submission_mca_tune <- data.frame(test_mca_jo_sum$USER_ID, pred_min_class_test_tune)
# colnames(submission_mca_tune) <- c("USER_ID", "Predictions")
# submission_mca_tune %>% glimpse()

## ------------- na_jo_sum_data ------------ ##
# Split the data into training and test set
set.seed(1398)
index <- train_na_jo_sum_final$Party %>% createDataPartition(p = .8, list = F)
train_na <- train_na_jo_sum_final[index,]
valid_na <- train_na_jo_sum_final[-index, ]
# data matrix 형태로 만들어주기!
train_na_x <- model.matrix(Party~., train_na)[,-1]
train_na_y <- train_na$Party

# 로지스틱 with lasso적합 alpha = 1
# 최적의 람다 값 탐색!
# Find the best lambda using cross-validation
set.seed(1398) 
cv.lasso <- cv.glmnet(train_na_x, train_na_y, alpha = 1, family = "binomial", type.measure = 'class', standardized = T)
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar = "lambda",label = F)

opt_lam_na <- cv.lasso$lambda.min # 오분류율 가장 낮게 하는 값
cv.lasso$lambda.1se # 표준오차가 가장 정규화되는 모델이 되게하는 람다 값

# 각 람다값에 따른 선택된 변수들
final_coef <- coef(cv.lasso, cv.lasso$lambda.min)
coef(cv.lasso, cv.lasso$lambda.1se)
rownames(final_coef)[which(final_coef!=0)]

# Final model with lambda.min
na_lasso_min <- glmnet(train_na_x, train_na_y, alpha = 1, family = "binomial", lambda = opt_lam_na)

# Make prediction on valid data
valid_na_x <- model.matrix(Party ~., valid_na)[,-1]
pred_na_min <- na_lasso_min %>% predict(newx = valid_na_x, type = 'response')

# 최적의 cut-off값 탐색!
pr_na_min <- prediction(pred_na_min, valid_na$Party)
prf_na_min <- performance(pr_na_min, measure = "tpr", x.measure = "fpr")

optid_na_min <- (1:length(prf_na_min@y.values[[1]][-1]))[((prf_na_min@x.values[[1]][-1])^2 + (1-prf_na_min@y.values[[1]][-11])^2)==min((prf_na_min@x.values[[1]][-1])^2 + (1-prf_na_min@y.values[[1]][-1])^2)]

# (0,1)에 가장 가까운 값 탐색!
optcut_na_min_x <- prf_na_min@x.values[[1]][-1][optid_na_min]
optcut_na_min_y <- prf_na_min@y.values[[1]][-1][optid_na_min]
optcut_na_min <- prf_na_min@alpha.values[[1]][-1][optid_na_min]

ggplot() +
  geom_point(aes(x = optcut_na_min_x, 
                 y = optcut_na_min_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_na_min_x, 
                y = optcut_na_min_y, label = prf_na_min@alpha.values[[1]][-1][optid_na_min] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_na_min@x.values[[1]],
                y = prf_na_min@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_min_class <- ifelse(pred_na_min >= optcut_na_min, 1, 0)

# Model accuracy
observed_class <- valid_na$Party
acc_na_min <- mean(pred_min_class == observed_class)

# Fit the model 단순 전체 적합이랑 비교!
full_model <- glm(Party ~., data = train_na, family = binomial)
# Make predictions
pred_full <- full_model %>% predict(valid_na, type = "response")

pr_full <- prediction(pred_full, valid_na$Party)
prf_full <- performance(pr_full, measure = "tpr", x.measure = "fpr")

optid_full <- (1:length(prf_full@y.values[[1]][-1]))[((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-11])^2)==min((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-1])^2)]

optcut_full_x <- prf_full@x.values[[1]][-1][optid_full]
optcut_full_y <- prf_full@y.values[[1]][-1][optid_full]
optcut_full <- prf_full@alpha.values[[1]][-1][optid_full]

ggplot() +
  geom_point(aes(x = optcut_full_x, 
                 y = optcut_full_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_full_x, 
                y = optcut_full_y, label = prf_full@alpha.values[[1]][-1][optid_full] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_full@x.values[[1]],
                y = prf_full@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_full_class <- ifelse(pred_full >= optcut_full, 1, 0)

# Model accuracy
observed_class <- valid_na$Party
acc_full <- mean(pred_full_class == observed_class)

# 그냥 다 적합했을 때 오히려 더 높은 정확도 보인다! acc_full

# 전체 적합해서 진행
# data matrix 형태로 만들어주기!
train_na_x <- model.matrix(Party~., train_na_jo_sum_final)[,-1]
train_na_y <- train_na_jo_sum_final$Party

na_lasso_min <- glmnet(train_na_x, train_na_y, alpha = 1, family = "binomial", lambda = opt_lam_na)

# test pred
test_na_x <- model.matrix(~.,test_na_jo_sum)[,-1]
pred_na_min_test <- na_lasso_min %>% predict(newx = test_na_x, type = 'response')
pred_min_class_test <- ifelse(pred_na_min_test >= optcut_na_min, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_na <- data.frame(test_na_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_na) <- c("USER_ID", "Predictions")
submission_na %>% glimpse()

# submission
fwrite(submission_na, file = "./submission/submission_na_final.csv", row.names = F) #

# Make prediction on test data
# test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
full_model <- glm(Party ~., data = train_na_jo_sum_final, family = binomial(link = "logit"))
pred_na_min_test <- full_model %>% predict(test_na_jo_sum, type = 'response')
pred_min_class_test <- ifelse(pred_na_min_test >= optcut_full, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_na <- data.frame(test_na_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_na) <- c("USER_ID", "Predictions")
submission_na %>% glimpse()

# submission
fwrite(submission_na, file = "./submission/submission_na_full.csv", row.names = F) # 

rm(list = setdiff(ls(), c("test_mca_jo_sum", "test_mean_jo_sum", "test_na_jo_sum", "test_nonanswer_jo_sum", "test_rf_jo_sum", "train_mca_jo_sum_final", "train_mean_jo_sum_final", "train_na_jo_sum_final", "train_nonanswer_jo_sum_final", "train_rf_jo_sum_final")))


## ------------- nonanswer_jo_sum_data ------------ ##
# Split the data into training and test set
set.seed(1398)
index <- train_nonanswer_jo_sum_final$Party %>% createDataPartition(p = .8, list = F)
train_nonanswer <- train_nonanswer_jo_sum_final[index,]
valid_nonanswer <- train_nonanswer_jo_sum_final[-index,]
# data matrix 형태로 만들어주기!
train_nonanswer_x <- model.matrix(Party~., train_nonanswer)[,-1]
train_nonanswer_y <- train_nonanswer$Party

# 로지스틱 with lasso적합 alpha = 1
# 최적의 람다 값 탐색!
# Find the best lambda using cross-validation
set.seed(1398) 
cv.lasso <- cv.glmnet(train_nonanswer_x, train_nonanswer_y, alpha = 1, family = "binomial", type.measure = 'class', standardized = T)
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar = "lambda",label = F, ylim = c(-1.5,1.5))
abline(v = log(opt_lam_nonanswer), lty = 2, col = "red")

opt_lam_nonanswer <- cv.lasso$lambda.min # 오분류율 가장 낮게 하는 값
cv.lasso$lambda.1se # 표준오차가 가장 정규화되는 모델이 되게하는 람다 값
log(0.0093)
# 각 람다값에 따른 선택된 변수들
coef(cv.lasso, cv.lasso$lambda.min)
coef(cv.lasso, cv.lasso$lambda.1se)

# Final model with lambda.min
nonanswer_lasso_min <- glmnet(train_nonanswer_x, train_nonanswer_y, alpha = 1, family = "binomial", lambda = opt_lam_nonanswer)
coef
# Make prediction on valid data
valid_nonanswer_x <- model.matrix(Party ~., valid_nonanswer)[,-1]
pred_nonanswer_min <- nonanswer_lasso_min %>% predict(newx = valid_nonanswer_x, type = 'response')

# 최적의 cut-off값 탐색!
pr_nonanswer_min <- prediction(pred_nonanswer_min, valid_nonanswer$Party)
prf_nonanswer_min <- performance(pr_nonanswer_min, measure = "tpr", x.measure = "fpr")

optid_nonanswer_min <- (1:length(prf_nonanswer_min@y.values[[1]][-1]))[((prf_nonanswer_min@x.values[[1]][-1])^2 + (1-prf_nonanswer_min@y.values[[1]][-11])^2)==min((prf_nonanswer_min@x.values[[1]][-1])^2 + (1-prf_nonanswer_min@y.values[[1]][-1])^2)]

# (0,1)에 가장 가까운 값 탐색!
optcut_nonanswer_min_x <- prf_nonanswer_min@x.values[[1]][-1][optid_nonanswer_min]
optcut_nonanswer_min_y <- prf_nonanswer_min@y.values[[1]][-1][optid_nonanswer_min]
optcut_nonanswer_min <- prf_nonanswer_min@alpha.values[[1]][-1][optid_nonanswer_min]

ggplot() +
  geom_point(aes(x = optcut_nonanswer_min_x, 
                 y = optcut_nonanswer_min_y),size = 3,  color = "red") +
  geom_text(aes(x = optcut_nonanswer_min_x, 
                y = optcut_nonanswer_min_y, label = prf_nonanswer_min@alpha.values[[1]][-1][optid_nonanswer_min] %>% round(3))
            ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
  geom_line(aes(x = prf_nonanswer_min@x.values[[1]],
                y = prf_nonanswer_min@y.values[[1]]), size = 1) +
  labs(x='FPR', y='TPR', title = 'ROC curve')

# 예측값 구하기
pred_min_class <- ifelse(pred_nonanswer_min >= optcut_nonanswer_min, 1, 0)

# Model accuracy
observed_class <- valid_nonanswer$Party
acc_nonanswer_min <- mean(pred_min_class == observed_class)
# 
# # Fit the model 단순 전체 적합이랑 비교!
# full_model <- glm(Party ~., data = train_nonanswer, family = binomial)
# # Make predictions
# pred_full <- full_model %>% predict(valid_nonanswer, type = "response")
# 
# pr_full <- prediction(pred_full, valid_nonanswer$Party)
# prf_full <- performance(pr_full, measure = "tpr", x.measure = "fpr")
# 
# optid_full <- (1:length(prf_full@y.values[[1]][-1]))[((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-11])^2)==min((prf_full@x.values[[1]][-1])^2 + (1-prf_full@y.values[[1]][-1])^2)]
# 
# optcut_full_x <- prf_full@x.values[[1]][-1][optid_full]
# optcut_full_y <- prf_full@y.values[[1]][-1][optid_full]
# optcut_full <- prf_full@alpha.values[[1]][-1][optid_full]
# 
# ggplot() +
#   geom_point(aes(x = optcut_full_x, 
#                  y = optcut_full_y),size = 3,  color = "red") +
#   geom_text(aes(x = optcut_full_x, 
#                 y = optcut_full_y, label = prf_full@alpha.values[[1]][-1][optid_full] %>% round(3))
#             ,hjust=-.1, check_overlap=TRUE, size = 5.5,  color = "red") +
#   geom_line(aes(x = prf_full@x.values[[1]],
#                 y = prf_full@y.values[[1]]), size = 1) +
#   labs(x='FPR', y='TPR', title = 'ROC curve')
# 
# # 예측값 구하기
# pred_full_class <- ifelse(pred_full >= optcut_full, 1, 0)
# 
# # Model accuracy
# observed_class <- valid_nonanswer$Party
# acc_full <- mean(pred_full_class == observed_class)
# 
# # full이 더 낫다

# 전체 적합해서 진행
# data matrix 형태로 만들어주기!
train_nonanswer_x <- model.matrix(Party~., train_nonanswer_jo_sum_final)[,-1]
train_nonanswer_y <- train_nonanswer_jo_sum_final$Party

nonanswer_lasso_min <- glmnet(train_nonanswer_x, train_nonanswer_y, alpha = 1, family = "binomial", lambda = opt_lam_nonanswer)

# test pred
test_nonanswer_x <- model.matrix(~.,test_nonanswer_jo_sum)[,-1]
pred_nonanswer_min_test <- nonanswer_lasso_min %>% predict(newx = test_nonanswer_x, type = 'response')
pred_min_class_test <- ifelse(pred_nonanswer_min_test >= optcut_nonanswer_min, 1, 0)
pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
submission_nonanswer <- data.frame(test_nonanswer_jo_sum$USER_ID, pred_min_class_test)
colnames(submission_nonanswer) <- c("USER_ID", "Predictions")
submission_nonanswer %>% glimpse()

# submission
fwrite(submission_nonanswer, file = "./submission/submission_nonanswer_final.csv", row.names = F) 

# 그냥 다 적합했을 때 오히려 더 높은 정확도 보인다! 
# 
# # Make prediction on test data
# # test_mca_x <- model.matrix(~.,test_mca_jo_sum)[,-1]
# full_model <- glm(Party ~., data = train_nonanswer_jo_sum_final, family = binomial(link = "logit"))
# pred_nonanswer_min_test <- full_model %>% predict(test_nonanswer_jo_sum, type = 'response')
# pred_min_class_test <- ifelse(pred_nonanswer_min_test >= optcut_full, 1, 0)
# pred_min_class_test <-  ifelse(pred_min_class_test==1, "Democrat", "Republican")
# submission_nonanswer <- data.frame(test_nonanswer_jo_sum$USER_ID, pred_min_class_test)
# colnames(submission_nonanswer) <- c("USER_ID", "Predictions")
# submission_nonanswer %>% glimpse()
# 
# # submission
# fwrite(submission_nonanswer, file = "./submission/submission_nonanswer_full.csv", row.names = F) # 
# 
# #최종 모델 Nonanswer + logistic with lasso 
# 
# set.seed(2020)
# opt_elastic <- train(~., train, method="glmnet", 
#                      trControl=trainControl("cv",number = 10), tuneLength=10)
# opt_elastic$bestTune
# opt_elastic
# y_pred3 <- predict(opt_elastic, test_x)
# RMSE(y_pred3, test_y)
