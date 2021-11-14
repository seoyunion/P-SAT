# ------ 변수 몽땅 지우기 ------ #
rm(list = ls())
# ------ basic package ------ #
library(tidyverse)
library(plyr)
library(magrittr)
library(data.table)
library(gridExtra)

# ------ 추가 package ------ #
library(readxl) # 엑셀 읽어오기
library(mice) # NA imputataion
library(GoodmanKruskal) 
library(rms) # describe()
library(lattice) # NA이후 시각화
library(missMDA)
library(FactoMineR)
library(factoextra)

# ------ choose working directory ------ #
setwd("C:/Users/user/Desktop/찬영/VISION/학회/P-SAT/주제분석/20년도 2학기 26기/주제분석 2-3주차")
getwd()

# ----- load all ------- #
load("C:/Users/user/Desktop/찬영/VISION/학회/P-SAT/주제분석/20년도 2학기 26기/주제분석 2-3주차/load_all_WEEK2.RData")

## ------ variable selection -------- ###
# Y vs X변수들 독립성 검정하는 함수
chisq_x_party <- function(data, alpha){
  chisq_test <- data.frame(colnames = colnames(data), p.value = rep(NA, ncol(data)), Hypothesis = rep(NA, ncol(data)))
  for(i in 1:ncol(data)){
    chisq_test[i,'p.value'] <- chisq.test(data[,colnames(data)[i]], data$Party)$p.value
    chisq_test[i,'Hypothesis'] <- ifelse(chisq_test[i,'p.value']<=alpha, 'reject_relate', 'independent')
  }
  return(chisq_test)
}

# X변수 간 독립성 검정하는 함수
chisq_x_x <- function(data, alpha){
  i=1
  chisq_test <- data.frame(x1 = rep(NA, choose(ncol(data),2)), x2 = rep(NA, choose(ncol(data),2)),
                           p.value = rep(NA, choose(ncol(data),2)), Hypothesis = rep(NA, choose(ncol(data),2)))
  for(j in 1:ncol(data)){
    for(k in 1:ncol(data)){
      chisq_test[i,'x1'] <- colnames(data)[j]
      chisq_test[i,'x2'] <- colnames(data)[k]
      chisq_test[i,'p.value'] <- chisq.test(data[,colnames(data)[j]], data[,colnames(data)[k]])$p.value
      chisq_test[i,'Hypothesis'] <- ifelse(chisq_test[i,'p.value']<=alpha, 'reject_relate', 'independent')
      i = i+1
    }
  }
  return(chisq_test)
}

## -------- DATA - train_rf_jo_sum 독립성 검정----------- ##
# x vs party -> 파생변수 중심으로 확인하자!
# life_sum, edu_sum 관련 있으면 질문들 잘가고.. 관련 없으면 파생변수 잘가고
rf_chisq_party <- chisq_x_party(train_rf_jo_sum, 0.05)
removelist_party <- rf_chisq_party %<>% filter(Hypothesis=="independent")

# Life_sum independent이므로 삭제!
train_rf_jo_sum %<>% select(-Life_sum)

## -------- GKtau로 연관성 확인하고, 제거 ------------- ##
find_sigpair = function(data) {
  Gktau = GKtauDataframe(data) %>% as.matrix()
  sig_pair = NULL
  for (i in 1:nrow(Gktau)) {
    for (j in 1:ncol(Gktau)) {
      if (i != j & Gktau[i,j] >= 0.5) {
        sig_pair = rbind(sig_pair,c(colnames(data)[i],colnames(data)[j]))
      }
    }
  }
  return(sig_pair)
}
gktau_list <- find_sigpair(train_rf_jo_sum[-1])
## 응..? 왜 이렇게 나와 질문간의 연관성은 없다..! marriage age householdstatus 아주 그냥 서로 연관있고.. 아마 glm에서 제거되지 않을까 하는..

## ------- 최종 train_rf_jo_sum_final 완성! ------- ##
train_rf_jo_sum_final <- train_rf_jo_sum
fwrite(train_rf_jo_sum_final, file =  "train_rf_jo_sum_final.csv", row.names = F)

## ------ test data 적용 ------- ##
test_Q_final <- fread(file ="test_Q_final.csv", header = T, stringsAsFactors = F, data.table = F)

# 김찬영 뭐했냐.. na 어떻게 처리할 건지 설정안해줘서 이거 추가로 해줘야 돼요.. ㅎㅎ..
for (i in 1:ncol(test_Q_final)) {
  colnames(test_Q_final)[i]
  test_Q_final[,i] <- replace(test_Q_final[,i], which(test_Q_final[,i]==""), NA)
}

test_Q_final %>% glimpse()

# ---------- DATA5: MICE with rf ------------------------ #
test_rf <- test_Q_final
# test는 Party 없으므로 반영해준다!
test_rf[,9:107] %<>% mutate_if(is.character, as.factor)
test_rf[,setdiff(feature,c("Income","EducationLevel", "Party"))] %<>% mutate_if(is.character, as.factor)
test_rf %>% glimpse()

# 질문들간의 연관 있는 것들로 NA를 채워보자!
mice_result_1rf <- mice(test_rf %>% select("mo_Q_fulltimejob","mo_Q_minwage_job",nonselect_Q[1:10]), seed=1234, m=1, method = 'rf')
complete_1rf <- complete(mice_result_1rf)

test_rf$mo_Q_fulltimejob <- complete_1rf$mo_Q_fulltimejob
test_rf$mo_Q_minwage_job <- complete_1rf$mo_Q_minwage_job
for (i in 1:10) {
  test_rf[,nonselect_Q[i]] <- complete_1rf[,nonselect_Q[i]]
}

mice_result_2rf <- mice(test_rf %>% select("Life_Q_collectHobby", "mo_Q_has_enoughcash_now","edu_Q_publicschool",nonselect_Q[11:20]), seed=1234, m=1, method = 'rf')
# 3개 연관된거랑 2개 연관된 거 또 다르네 시간이 있으면 둘이 비교해보고 아니면 3개로 넣자
complete_2rf <- complete(mice_result_2rf)
test_rf$Life_Q_collectHobby <- complete_2rf$Life_Q_collectHobby
test_rf$mo_Q_has_enoughcash_now <- complete_2rf$mo_Q_has_enoughcash_now
test_rf$edu_Q_publicschool <- complete_2rf$edu_Q_publicschool
for (i in 11:20) {
  test_rf[,nonselect_Q[i]] <- complete_2rf[,nonselect_Q[i]]
}

mice_result_3rf <- mice(test_rf %>% select("Life_Q_drink", "re_Q_newromance",nonselect_Q[21:30]), seed=1234, m=1, method = 'rf')
complete_3rf <- complete(mice_result_3rf)
test_rf$Life_Q_drink <- complete_3rf$Life_Q_drink
test_rf$re_Q_newromance <- complete_3rf$re_Q_newromance
for (i in 21:30) {
  test_rf[,nonselect_Q[i]] <- complete_3rf[,nonselect_Q[i]]
}

mice_result_4rf <- mice(test_rf %>% select( "Life_Q_gun", "ps_Q_PowerOfPositive",nonselect_Q[31:40]), seed=1234, m=1, method = 'rf')
complete_4rf <- complete(mice_result_4rf)
test_rf$Life_Q_gun <- complete_4rf$Life_Q_gun
test_rf$ps_Q_PowerOfPositive <- complete_4rf$ps_Q_PowerOfPositive
for (i in 31:40) {
  test_rf[,nonselect_Q[i]] <- complete_4rf[,nonselect_Q[i]]
}

mice_result_5rf <- mice(test_rf %>% select("edu_Q_parents_college","env_Q_single_parent",nonselect_Q[41:50]), seed=1234, m=1, method = 'rf')
complete_5rf <- complete(mice_result_5rf)
test_rf$edu_Q_parents_college <- complete_5rf$edu_Q_parents_college
test_rf$env_Q_single_parent <- complete_5rf$env_Q_single_parent
for (i in 41:50) {
  test_rf[,nonselect_Q[i]] <- complete_5rf[,nonselect_Q[i]]
}

mice_result_6rf <- mice(test_rf %>% select("re_Q_likepeople", "ps_Q_LikeFirstName", nonselect_Q[51:60]), seed=1234, m=1, method = 'rf')
complete_6rf <- complete(mice_result_6rf)
test_rf$re_Q_likepeople <- complete_6rf$re_Q_likepeople
test_rf$ps_Q_LikeFirstName <- complete_6rf$ps_Q_LikeFirstName
for (i in 51:60) {
  test_rf[,nonselect_Q[i]] <- complete_6rf[,nonselect_Q[i]]
}

mice_result_7rf <- mice(test_rf %>% select("ps_Q_GoodLiar", "Life_Q_work50h", nonselect_Q[61:70]), seed=1234, m=1, method = 'rf')
complete_7rf <- complete(mice_result_7rf)
test_rf$ps_Q_GoodLiar <- complete_7rf$ps_Q_GoodLiar
test_rf$Life_Q_work50h <- complete_7rf$Life_Q_work50h
for (i in 61:70) {
  test_rf[,nonselect_Q[i]] <- complete_7rf[,nonselect_Q[i]]
}

mice_result_8rf <- mice(test_rf %>% select("Life_Q_watchTV","env_Q_p_spank",nonselect_Q[71:78]), seed=1234, m=1, method = 'rf')
complete_8rf <- complete(mice_result_8rf)
test_rf$Life_Q_watchTV <- complete_8rf$Life_Q_watchTV
test_rf$env_Q_p_spank <- complete_8rf$env_Q_p_spank
for (i in 71:78) {
  test_rf[,nonselect_Q[i]] <- complete_8rf[,nonselect_Q[i]]
}

mice_result_9rf <- mice(test_rf %>% select("Life_Q_livealone", "ps_Q_LeftHanded", sample(nonselect_Q,10)), seed=1234, m=1, method = 'rf')
complete_9rf <- complete(mice_result_9rf)
test_rf$Life_Q_livealone <- complete_9rf$Life_Q_livealone
test_rf$ps_Q_LeftHanded <- complete_9rf$ps_Q_LeftHanded

mice_result_10rf <- mice(test_rf %>% select("ps_Q_Creative", "re_Q_havesibling", sample(nonselect_Q,10)), seed=1234, m=1, method = 'rf')
complete_10rf <- complete(mice_result_10rf)
test_rf$ps_Q_Creative <- complete_10rf$ps_Q_Creative
test_rf$re_Q_havesibling <- complete_10rf$re_Q_havesibling
test_rf %>% is.na() %>% colSums()

# ------------- 새롭게 만든 데이터에 조나단 스코어링 진행 ------- #
test_rf_jo <- test_rf
test_rf_jo$careness = jonathan('careness', test_rf)
test_rf_jo$fairness = jonathan('fairness', test_rf)
test_rf_jo$loyalty = jonathan('loyalty', test_rf)
test_rf_jo$authority = jonathan('authority',test_rf)
test_rf_jo$sanctity = jonathan('sanctity', test_rf)

# ----------- 새롭게 만든 sum파생변수 적용 ------------ #
test_rf_jo_sum <- test_rf_jo

# 제외
# test_rf_jo_sum$Life_sum = ifelse(train_rf_jo_sum$Life_Q_exerc3we == "Yes",1,0) +
#   ifelse(train_rf_jo_sum$Life_Q_breakfast == "Yes",1,0) +
#   ifelse(train_rf_jo_sum$Life_Q_earlyAlarm == "Yes",1,0) +
#   ifelse(train_rf_jo_sum$Life_Q_standardTime == "Yes",1,0)

test_rf_jo_sum$edu_sum = ifelse(test_rf_jo_sum$edu_Q_allA == "Yes",1,0) +
  ifelse(test_rf_jo_sum$edu_Q_goodat_math == "Yes",1,0) +
  ifelse(test_rf_jo_sum$edu_Q_parents_college == "Yes",1,0) +
  ifelse(test_rf_jo_sum$edU_Q_mas_doc_degree == "Yes",1,0)

test_rf_jo_sum %>% glimpse()

fwrite(test_rf_jo_sum, file = "test_rf_jo_sum.csv", row.names = F)

# ----- train / test 범주 개수 비교 ----- #
compare_lv = NULL
for (col in colnames(test_rf_jo_sum)) {
  compare_lv = rbind(compare_lv,c(train_rf_jo_sum_final[,col] %>% n_distinct, test_rf_jo_sum[,col] %>% n_distinct))
}
compare_lv

colnames(test_rf_jo_sum)