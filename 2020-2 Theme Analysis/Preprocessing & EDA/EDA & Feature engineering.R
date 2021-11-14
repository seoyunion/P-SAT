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

# ------ Data load ------ #
train <- fread(file = "./can-we-predict-voting-outcomes/train2016.csv", header = T, stringsAsFactors = F, data.table = F)
test <- fread(file =  "./can-we-predict-voting-outcomes/test2016.csv", header = T, stringsAsFactors = F, data.table = F)

# ----- Change variable name ----- #
# 수정이 part
category_info = read_excel('설문조사 질문 정리_카테고리+역할배분.xlsx',sheet=1,col_names = TRUE)
category_info = category_info %>% select(`Question ID`,`문제`,`범주 Category`)
colnames(category_info) = c('Q_ID','Q','category')

new_Qname = category_info %>% 
  filter(category %in% c('Relationships','Money','Education')) %>% 
  mutate(new_Q = c('re_Q_havesibling','re_Q_Dad_householdpower','re_Q_carrygrudge','re_Q_likepeople','re_Q_meetoffline','re_Q_extendedfamilly','re_Q_betterlookthan_friend','re_Q_successfulthan_friend','re_Q_newromance','re_Q_jealous','re_Q_interact_dislike',
                   'mo_Q_own_residence','mo_Q_has_debt','mo_Q_feel_financialdebt','mo_Q_poor','mo_Q_live_metro','mo_Q_carpayment','mo_Q_has_enoughcash_now','mo_Q_minwage_job','mo_Q_fulltimejob',
                   'edu_Q_goodat_math','edu_Q_parents_college','edu_Q_allA','edu_Q_mas/doc_degree','edu_Q_publicschool'
  ))

rm(category_info)

for (i in 1:ncol(train)) {
  A = sub('Q','',colnames(train)[i])
  
  for (j in 1:nrow(new_Qname)) {
    B = new_Qname[j,'Q_ID'] %>% as.character()
    
    if (A == B) colnames(train)[i] = new_Qname[j,'new_Q']
  }
}

train = train %>% 
  mutate(re_Q_havesibling = mapvalues(re_Q_havesibling,from = c('Only-child','Yes'),to = c('No','Yes')),
         re_Q_Dad_householdpower = mapvalues(re_Q_Dad_householdpower,from = c('Mom','Dad'),to = c('No','Yes')),
         re_Q_likepeople = mapvalues(re_Q_likepeople,from = c('Yay people!','Grrr people'),to = c('Yes','No')),
         re_Q_meetoffline = mapvalues(re_Q_meetoffline,from = c('In-person','Online'),to = c('Yes','No')),
         re_Q_extendedfamilly = mapvalues(re_Q_extendedfamilly,from = c('Yes!','Umm...'),to = c('Yes','No')),
         mo_Q_own_residence = mapvalues(mo_Q_own_residence,from = c('Own','Rent'),to = c('Yes','No')),
         edu_Q_publicschool = mapvalues(edu_Q_publicschool,from = c('Public','Private'),to = c('Yes','No')))

rm(new_Qname)

train %>% colnames()

# 서윤이 part
lifestyle <- c("Q113181","Q98578","Q99716","Q100010","Q102687","Q103293","Q104996","Q105655","Q106272","Q106388","Q110740","Q111220",
               "Q113583","Q113584","Q113992","Q114517","Q114748","Q115611","Q116197","Q117193","Q118892","Q119851","Q121699","Q122769")

name <- c("Life_Q_medipray","Life_Q_exerc3we","Life_Q_livealone","Life_Q_watchTV","Life_Q_breakfast","Life_Q_pet","Life_Q_brushT2",
          "Life_Q_morningAlarm","Life_Q_ownTool","Life_Q_work50h","Life_Q_MacPC","Life_Q_earlyAlarm","Life_Q_driveTunesTalk",
          "Life_Q_PeopleTech","Life_Q_gamble","Life_Q_morningTV","Life_Q_tabwater","Life_Q_gun","Life_Q_AMPM","Life_Q_standardTime",
          "Life_Q_glasses","Life_Q_readBook","Life_Q_drink","Life_Q_collectHobby")

for (i in 1:ncol(train)){
  for (k in 1:length(lifestyle)){
    if (colnames(train)[i] == lifestyle[k]){
      colnames(train)[i] = name[k]
    }
  }
}

train = train %>% 
  mutate(Life_Q_standardTime = mapvalues(Life_Q_standardTime,from = c('Standard hours','Odd hours'),to = c('Yes','No')),
         Life_Q_driveTunesTalk = mapvalues(Life_Q_driveTunesTalk,from = c("Talk",'Tunes'),to = c('No','Yes')),
         Life_Q_PeopleTech = mapvalues(Life_Q_PeopleTech,from = c("Technology",'People'),to = c('No','Yes')),
         Life_Q_AMPM = mapvalues(Life_Q_AMPM, from = c("A.M.","P.M."), to = c("Yes",'No')),
         Life_Q_MacPC = mapvalues(Life_Q_MacPC,from = c('Mac','PC'),to = c("Yes","No")))

rm(lifestyle)
rm(name)


# health category Question name 변경
health <- c("Q106042","Q116797")
name <- c("Hel_Q_medicineTake","Hel_Q_dailyVitamin")

for (i in 1:ncol(train)){
  for (k in 1:length(health)){
    if (colnames(train)[i] == health[k]){
      colnames(train)[i] = name[k]
    }
  }
}

rm(health)
rm(name)

train %>% colnames()

# 은주 part
personality <- c("Q98078", "Q98869", "Q99581", "Q99982", "Q100562", "Q100689", "Q101162", "Q102289", "Q105840", "Q106389", "Q106993", "Q107491", "Q107869", "Q108856", "Q108950", "Q109244", "Q112478", "Q112512", "Q114152", "Q114386", "Q114961", "Q115390", "Q115602", "Q115610", "Q115777", "Q115899", "Q116448", "Q116881", "Q116953", "Q117186", "Q118232", "Q119650", "Q120012", "Q120194", "Q120472")
name <- c("ps_Q_Creative", "ps_Q_LifePurpose", "ps_Q_LeftHanded", "ps_Q_Checklist", "ps_Q_BetterAfter5y", "ps_Q_Overweight", "ps_Q_Optimist", "ps_Q_Adventurous", "ps_Q_RetailTherapy", "ps_Q_GoodLiar", "ps_Q_LikeFirstName", "ps_Q_Punctuate", "ps_Q_Normal", "ps_Q_Socializing", "ps_Q_RiskFriendly", "ps_Q_Feminist", "ps_Q_Phobia", "ps_Q_Skeptical", "ps_Q_SupportCharity", "ps_Q_OverShare", "ps_Q_BuyHappiness","ps_Q_ChangedPersonality", "ps_Q_Obedient", "ps_Q_PowerOfPositive", "ps_Q_StartHabit", "ps_Q_ResultOfCircumstance", "ps_Q_NoLieChangeLife", "ps_Q_happy>right", "ps_Q_Rules", "ps_Q_QuickTemper", "ps_Q_Idealist", "ps_Q_Giving>Receiving", "ps_Q_WeatherMood", "ps_Q_Study>Try", "ps_Q_Science>Art")

for (i in 1:ncol(train)){
  for (k in 1:length(personality)){
    if (colnames(train)[i] == personality[k]){
      colnames(train)[i] = name[k]
    }
  }
}

rm(personality)
rm(name)

# 답변 Y/N로 통일
train$`ps_Q_Science>Art` <- ifelse(train$`ps_Q_Science>Art` == "Science", "Yes", "No")
train$ps_Q_Checklist <- ifelse(train$ps_Q_Checklist == "Check!", "Yes", "No")
train$ps_Q_Optimist <- ifelse(train$ps_Q_Optimist == "Optimist", "Yes", "No")
train$ps_Q_Socializing <- ifelse(train$ps_Q_Socializing == "Socialize", "Yes", "No")
train$ps_Q_RiskFriendly <- ifelse(train$ps_Q_RiskFriendly == "Risk-friendly", "Yes", "No")
train$ps_Q_OverShare <- ifelse(train$ps_Q_OverShare == "TMI", "Yes", "No")
train$ps_Q_StartHabit <- ifelse(train$ps_Q_StartHabit == "Start", "Yes", "No")
train$ps_Q_ResultOfCircumstance <- ifelse(train$ps_Q_ResultOfCircumstance == "Circumstances", "Yes", "No")
train$`ps_Q_happy>right` <- ifelse(train$`ps_Q_happy>right` == "Happy", "Yes", "No")
train$ps_Q_QuickTemper <- ifelse(train$ps_Q_QuickTemper == "Hot headed", "Yes", "No")
train$ps_Q_Idealist <- ifelse(train$ps_Q_Idealist == "Idealist", "Yes", "No")
train$`ps_Q_Giving>Receiving` <- ifelse(train$`ps_Q_Giving>Receiving` == "Giving", "Yes", "No")
train$`ps_Q_Study>Try` <- ifelse(train$`ps_Q_Study>Try` == "Study first", "Yes", "No")

train %>% colnames()

# 혜인 part
# Experience category Question name 변경
experience <- c("Q100680", "Q101596", "Q116601", "Q118233", "Q119334", "Q121011")
name <- c("ex_Q_cry60D", "ex_Q_treehouse", "ex_Q_travelabroad", "ex_Q_crimethreatened","ex_Q_accomplish","ex_Q_stress_event" )

for (i in 1:ncol(train)){
  for (k in 1:length(experience)){
    if (colnames(train)[i] == experience[k]){
      colnames(train)[i] = name[k]
    }
  }
}

rm(experience)
rm(name)

# Environ category Question name 변경

environ <- c("Q99480", "Q108617", "Q111580", "Q118117", "Q118237", "Q120650","Q120978","Q124122")
name <- c("env_Q_p_spank", "env_Q_single_parent", "env_Q_p_sup_dem", "env_Q_same_state","env_Q_overhead_task","env_Q_p_married","env_Q_Sesame_str","env_Q_p_fight.ifoU" )

for (i in 1:ncol(train)){
  for (k in 1:length(environ)){
    if (colnames(train)[i] == environ[k]){
      colnames(train)[i] = name[k]
    }
  }
}

rm(environ)
rm(name)

#Y/N로 바꾸기

train = train %>% mutate(env_Q_p_sup_dem = mapvalues(env_Q_p_sup_dem,from = c('Supportive','Demanding'),to = c('Yes','No')))

train %>% colnames()

# ---------- 질문명 직관적으로 변경 + 모든 질문 Yes / no로 치환 -------- #
# ------- "" -> NA 바꾸기 ------- #
for (i in 1:ncol(train)) {
  colnames(train)[i]
  train[,i] <- replace(train[,i], which(train[,i]==""), NA)
}
rm(i,A,B,j,k)
for (i in 1:ncol(test)) {
  colnames(test)[i]
  test[,i] <- replace(test[,i], which(test[,i]==""), NA)
}
rm(i,A,B,j,k)
train %>% is.na() %>% mean()
test %>% is.na() %>% mean()
# 변수 명 바뀐 데이터셋 확인
train %>% str()
train %>% glimpse()
train %>% head()
train %>% tail()
train %>% summary()
train %>% colnames()

# Q98197과 Q113181은 LIfe_Q_medipray로 동일한 문항 NA 더 많은 Q98197 제외하자!
train$Q98197 %>% is.na() %>% sum()
train$Life_Q_medipray %>% is.na() %>% sum()

train_Q <- train %>% select(-Q98197)

# ------ 변수 명 및 yes or no 바뀐 파일 저장 ----- #
write.csv(train_Q, file = 'train_Q.csv', row.names = F)

# train_Q
# 1. 변수명 직관적으로 변경
# 2. 데이터 핸들링 쉽도록 "" -> NA로 변경
# 3. 중복 문항(주기적으로 명상이나 기도?) NA 더 적은 것만 남김

train_1 <- train_Q
# 인적 변수 인코딩
# YOB -> 2014년 기준 나이로 변경
train_1$YOB <- 2014-as.numeric(train$YOB)

# 나이 이상치 제거
UpperOutlier_age <- which(train_1$YOB >= 100)
LowerOutlier_age <- which(train_1$YOB <= 3)
age_outlier_index <- c(UpperOutlier_age, LowerOutlier_age)
train_1 <- train_1[-age_outlier_index, ]

# NA비율 50% 이상 질문 제거
train_1 %<>% select(-re_Q_interact_dislike)
# 이번엔 User 별로 NA 비율을 보고 삭제할만한 row를 탐색해보자
na.count <- function(x){sum(is.na(x))}
na.prop <- apply(train_1,2,na.count)
na_prop <- data.frame(col = names(na.prop), NA.prop = round(na.prop/nrow(train),2)*100) %>% dplyr::arrange(desc(na.prop))
train_question <- train_1[,c(1,8:106)]
na.prop_user <- apply(train_question,1,na.count)
na_prop_user <- data.frame(user = train_question$USER_ID, NA.prop_user = round(na.prop_user/ncol(train_question),2)*100) %>% dplyr::arrange(desc(na.prop_user))

#80% 이상 무응답자 제거 의논 이후 %만 바꾸면 됨
removelist_user <- na_prop_user %>% filter(NA.prop_user>=80) %>% select(user)
removelist_user <- removelist_user$user
train_1 <- train_1 %>% filter(USER_ID%in%setdiff(train$USER_ID,removelist_user))

# train_1
# 1. 무응답 비율 80% 넘은 user 제거 
# 2. 문항 기준 무응답 비율 50% 넘는 항목 re_Q_interact_dislike 제거
# 3. 나이 이상치 제거

# -------------- 인적변수 Encoding 및 파생변수 생성 --------------- #
train_2 <- train_1
train_2 %>% glimpse()

# Party factor로 치환
#train_2$Party <- ifelse(train_2$Party=='Democrat', 1, 0)

## --- YOB 변수 --- ##
# YOB 변수 -> 연령대 변수 age
train_2 <- train_2 %>% mutate(age = ifelse(train_2$YOB <= 19, "10s", ifelse(train_2$YOB <= 29, "20s", ifelse(train_2$YOB <= 39, "30s", ifelse(train_2$YOB <= 49, "40s", ifelse(train_2$YOB <= 59, "50s", ifelse(train_2$YOB <= 69, "60s", ifelse(train_2$YOB <= 79, "70s", "80s"))))))))

train_2 %<>% select(-YOB) # YOB 변수 삭제


## --- Income 변수 --- ##
# Ordinal Encoding
train_2$Income %<>% 
  revalue(c("under $25,000" = 0.18, "$25,001 - $50,000" = 0.43, "$50,000 - $74,999" = 0.61, "$75,000 - $100,000" = 0.74,
            "$100,001 - $150,000" = 0.88,"over $150,000" =1 )) %<>% as.numeric()


## --- HousholdStatus 변수 --- ##
# 혼인 상태 (married, Domestic, Single)
train_2 <- train_2 %>% mutate(marriage = ifelse(HouseholdStatus == "Married (w/kids)" | HouseholdStatus == "Married (no kids)", "Married", ifelse(HouseholdStatus == "Domestic Partners (w/kids)" | HouseholdStatus == "Domestic Partners (no kids)", "Domestic Partners", "Single")))

# 자식 여부
train_2 <- train_2 %>% mutate(kids = ifelse(HouseholdStatus == "Married (w/kids)" | HouseholdStatus == "Domestic Partners (w/kids)" | HouseholdStatus == "Signle (w/kids)", "w/kids", "no kids"))

## --- EducationLevel 변수 --- ##
# 학부 수준을 기준으로 재범주화 및 Ordinal Encoding
train_2$EducationLevel %<>% 
  revalue(c("Current K-12" = 1, "High School Diploma" = 1, "Current Undergraduate" = 2, "Associate's Degree" = 2, "Bachelor's Degree" = 2,
            "Master's Degree" = 3, "Doctoral Degree" = 3)) %<>% as.numeric()

feature <- c("USER_ID","Gender","Income","HouseholdStatus","EducationLevel","Party","age","marriage","kids")
# 인적변수들 무응답 범주 추가
for (i in feature){
  train_2[,i] = replace(train_2[,i], which(is.na(train_2[,i])), "non_answer")
}

# Income, Education 무응답 범주포함하여 ordinal 처리
# 소득 무응답 = 0 , education 무응답 = 0
train_2$Income %<>% revalue(c("non_answer"=0)) %>% as.numeric()
train_2$EducationLevel %<>% revalue(c("non_answer"=0)) %>% as.numeric()

## ----- factor 화 --- ##
# train_2$Gender <- train_2$Gender %>% as.factor()
# train_2$age %<>% as.factor()
# train_2$EducationLevel %<>% as.factor()
# train_2$marriage %<>% as.factor()
# train_2$kids %<>% as.factor()
# train_2$HouseholdStatus %<>% as.factor()
# train_2$Income %<>% as.factor()
#train_2$Party %<>% as.factor()

# ------ 설문문항 인코딩 ------ #
#train_2[,10:108] %<>% mutate_if(is.character, as.factor)

# 미리 factor 하니 이후에 많이 꼬여서 모델링 직전에 dataset 조절합시다!
# ordinal -> Income(소득 분위), EducationLevel(교육 수준),
# factor -> Gender(성별), HouseHoldStatus(가구형태), Party, age(나이대), marriage(기혼 여부), kids(자식유무)

# 한번에 바꾼 변수들 쉽게 열 순서 조정
train_2 <- train_2[,c(1:6,106:108,7:105)]
train_2 %>% colnames()

train_2 %>% str()
train_2 %>% glimpse()
 

#train_2
# 1. 인적변수 NA값들 non-answer로 치환 및 인적변수 인코딩과 재범주화 

# ----- Feature Engineering ----- #
train_3 <- train_2
compute_score = function(row,col,type,data) {
  if (!is.na(data[row,col])) {
    if (type == 'careness') {
      if (col %in% c('ps_Q_GoodLiar','ps_Q_Feminist','ps_Q_SupportCharity','ps_Q_ResultOfCircumstance')) {
        if (data[row,col] == "Yes") return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'fairness') {
      if (col %in% c('mo_Q_minwage_job','edu_Q_parents_college','edu_Q_publicschool','ps_Q_Optimist','ps_Q_GoodLiar','ps_Q_Normal','ps_Q_Feminist','ps_Q_SupportCharity','ps_Q_ResultOfCircumstance')) {
        if (data[row,col] == "Yes") return(1)
        else return(0)
      }
      else if (col %in% c('ps_Q_BuyHappiness','ps_Q_happy>right')) {
        if (data[row,col] == "No") return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'loyalty') {
      if (col %in% c('ps_Q_LikeFirstName','ps_Q_Obedient','ps_Q_Idealist','env_Q_same_state','ps_Q_RiskFriendly')) {
        if (data[row,col] == "Yes") return(1)
        else return(0)
      }
      else if (col %in% c('ps_Q_Idealist')) {
        if (data[row,col] == "No") return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'authority') {
      if (col %in% c('re_Q_successfulthan_friend','mo_Q_own_residence','mo_Q_live_metro','mo_Q_has_enoughcash_now','mo_Q_fulltimejob','edu_Q_parents_college','edu_Q_mas/doc_degree','Life_Q_gun','ps_Q_BuyHappiness','ps_Q_Obedient','ps_Q_happy>right','env_Q_p_sup_dem')) {
        if (data[row,col] == "Yes") return(1)
        else return(0)
      }
      else if (col %in% c('mo_Q_has_debt','mo_Q_feel_financialdebt','mo_Q_poor','mo_Q_carpayment','mo_Q_minwage_job','edu_Q_publicschool','ps_Q_Feminist','ps_Q_Idealist')) {
        if (data[row,col] == "No") return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'sanctity') {
      if (col %in% c('Life_Q_medipray','ps_Q_happy>right','ps_Q_RiskFriendly')) {
        if (data[row,col] == "Yes") return(1)
        else return(0)
      }
      else if (col %in% c('edu_Q_publicschool')) {
        if (data[row,col] == "Yes") return(1)
        else return(0)
      }
      else return(0)
    }
    else return(0)
  }
  else return(0)
}

jonathan = function(type, data) {
  new_col = rep(NA,nrow(data))
  cols = colnames(data)[grep('Q_',colnames(data))]
  
  for (i in 1:nrow(data)) {
    score = 0
    
    for (col in cols) {
      score = score + compute_score(i,col,type,data)
    }
    new_col[i] = score
  }
  return(new_col)
}
train_3$careness = jonathan('careness', train_2)
train_3$fairness = jonathan('fairness', train_2)
train_3$loyalty = jonathan('loyalty', train_2)
train_3$authority = jonathan('authority', train_2)
train_3$sanctity = jonathan('sanctity', train_2)


# 튕기는 애들 탐색
colnames(train_3)
revise1 <- grep('>',colnames(train_3))
colnames(train_3)[revise1] <- c("ps_Q_Science_Art","ps_Q_Study_Try","ps_Q_Giving_Receiving","ps_Q_happy_right")
revise2 <- grep("/", colnames(train_3))
colnames(train_3)[revise2] <- c("edU_Q_mas_doc_degree")

train_3 %>% glimpse()
fwrite(train_3, file = 'train_EDA.csv', row.names = F, na = NA)

#train_3
# 조나단 파생변수 추가
# MICE 들어가기 전 튕기는 열 이름 변경


# -------------------DATA1: MICE with mean ------------------------ #
# train_mean은 MICE에 들어갈 Inqut data
train_mean <- train_3
# train_4에는 MICE를 거치고 채워질 것들..
train_4 <- train_3

# 알고리즘 변경 -> 인적변수는 무응답 범주 생김 나머지 질문 변수만 cutoff=0.5적용
train_mean[,10:108] %<>% mutate_if(is.character, as.factor)
train_mean[,10:108] %<>% mutate_if(is.factor, as.integer)
train_mean[,10:108] <- train_mean[,10:108]-1
train_mean %>% glimpse()
for (i in 10:108) {
    train_mean[,i] = replace(train_mean[,i], which(is.na(train_mean[,i])), ifelse(mean(train_mean[,i], na.rm = T)>=0.5,1,0)) }

colSums(is.na(train_mean))

# 결측값 채운 후, 모델링 이전 데이터 구조 조정 로지스틱 사용하기 위해 factor화
# 질문 factor로
train_mean[, 10:108] %<>% mutate_if(is.numeric, as.factor)

# 인적변수 factor로
train_mean[,setdiff(feature,c("Income","EducationLevel"))] %<>% mutate_if(is.character, as.factor)
train_mean %>% glimpse()


## ---------- imputation 시작 -------------------- ###
# 질문들간의 연관 있는 것들로 NA를 채워보자! -> 알고리즘 짝들
select_Q <- c("mo_Q_fulltimejob","mo_Q_minwage_job","Life_Q_collectHobby", "mo_Q_has_enoughcash_now","edu_Q_publicschool","Life_Q_drink", "re_Q_newromance", "Life_Q_gun", "ps_Q_PowerOfPositive", "edu_Q_parents_college","env_Q_single_parent","re_Q_likepeople", "ps_Q_LikeFirstName", "Life_Q_watchTV","env_Q_p_spank","Life_Q_livealone", "ps_Q_LeftHanded", "ps_Q_GoodLiar" , "Life_Q_work50h","ps_Q_Creative","re_Q_havesibling")
length(select_Q)
# 상관계수에서 선택받지 못한 자들..
nonselect_Q <- setdiff(colnames(train_mean),select_Q)
nonselect_Q <- nonselect_Q[-c(1:9,88:92)]

# 공부해보니 변수 2개로는 너무 적고 15~25개의 설문을 활용한다고 돼있음! 2개 이외의 선택되지 않은 설문을 추가하여 변수 생성 진행 상관계수 높은거부터 차례로 나열할라 했으나 너무 복잡..
# 알고리즘 변경 X-Y 높은것만 남기고 나머지는 상관성 없는 걸로 넣어서 진행
mice_result_1 <- mice(train_mean %>% select("mo_Q_fulltimejob","mo_Q_minwage_job",nonselect_Q[1:10]), seed=1234, m=5)
complete_1 <- complete(mice_result_1)
train_4$mo_Q_fulltimejob <- complete_1$mo_Q_fulltimejob
train_4$mo_Q_minwage_job <- complete_1$mo_Q_minwage_job
for (i in 1:10) {
  train_4[,nonselect_Q[i]] <- complete_1[,nonselect_Q[i]]
}

mice_result_2 <- mice(train_mean %>% select("Life_Q_collectHobby", "mo_Q_has_enoughcash_now","edu_Q_publicschool",nonselect_Q[11:20]), seed=1234, m=5)
# 3개 연관된거랑 2개 연관된 거 또 다르네 시간이 있으면 둘이 비교해보고 아니면 3개로 넣자
complete_2 <- complete(mice_result_2)
train_4$Life_Q_collectHobby <- complete_2$Life_Q_collectHobby
train_4$mo_Q_has_enoughcash_now <- complete_2$mo_Q_has_enoughcash_now
train_4$edu_Q_publicschool <- complete_2$edu_Q_publicschool
for (i in 11:20) {
  train_4[,nonselect_Q[i]] <- complete_2[,nonselect_Q[i]]
}

mice_result_3 <- mice(train_mean %>% select("Life_Q_drink", "re_Q_newromance",nonselect_Q[21:30]), seed=1234, m=5)
complete_3 <- complete(mice_result_3)
train_4$Life_Q_drink <- complete_3$Life_Q_drink
train_4$re_Q_newromance <- complete_3$re_Q_newromance
for (i in 21:30) {
  train_4[,nonselect_Q[i]] <- complete_3[,nonselect_Q[i]]
}

mice_result_4 <- mice(train_mean %>% select( "Life_Q_gun", "ps_Q_PowerOfPositive",nonselect_Q[31:40]), seed=1234, m=5)
complete_4 <- complete(mice_result_4)
train_4$Life_Q_gun <- complete_4$Life_Q_gun
train_4$ps_Q_PowerOfPositive <- complete_4$ps_Q_PowerOfPositive
for (i in 31:40) {
  train_4[,nonselect_Q[i]] <- complete_4[,nonselect_Q[i]]
}

mice_result_5 <- mice(train_mean %>% select("edu_Q_parents_college","env_Q_single_parent",nonselect_Q[41:50]), seed=1234, m=5)
complete_5 <- complete(mice_result_5)
train_4$edu_Q_parents_college <- complete_5$edu_Q_parents_college
train_4$env_Q_single_parent <- complete_5$env_Q_single_parent
for (i in 41:50) {
  train_4[,nonselect_Q[i]] <- complete_5[,nonselect_Q[i]]
}

mice_result_6 <- mice(train_mean %>% select("re_Q_likepeople", "ps_Q_LikeFirstName", nonselect_Q[51:60]), seed=1234, m=5)
complete_6 <- complete(mice_result_6)
train_4$re_Q_likepeople <- complete_6$re_Q_likepeople
train_4$ps_Q_LikeFirstName <- complete_6$ps_Q_LikeFirstName
for (i in 51:60) {
  train_4[,nonselect_Q[i]] <- complete_6[,nonselect_Q[i]]
}

mice_result_7 <- mice(train_mean %>% select("ps_Q_GoodLiar", "Life_Q_work50h", nonselect_Q[61:70]), seed=1234, m=5)
complete_7 <- complete(mice_result_7)
train_4$ps_Q_GoodLiar <- complete_7$ps_Q_GoodLiar
train_4$Life_Q_work50h <- complete_7$Life_Q_work50h
for (i in 61:70) {
  train_4[,nonselect_Q[i]] <- complete_7[,nonselect_Q[i]]
}

mice_result_8 <- mice(train_mean %>% select("Life_Q_watchTV","env_Q_p_spank",nonselect_Q[71:78]), seed=1234, m=5)
complete_8 <- complete(mice_result_8)
train_4$Life_Q_watchTV <- complete_8$Life_Q_watchTV
train_4$env_Q_p_spank <- complete_8$env_Q_p_spank
for (i in 71:78) {
  train_4[,nonselect_Q[i]] <- complete_8[,nonselect_Q[i]]
}

mice_result_9 <- mice(train_mean %>% select("Life_Q_livealone", "ps_Q_LeftHanded", sample(nonselect_Q,10)), seed=1234, m=5)
complete_9 <- complete(mice_result_9)
train_4$Life_Q_livealone <- complete_9$Life_Q_livealone
train_4$ps_Q_LeftHanded <- complete_9$ps_Q_LeftHanded

mice_result_10 <- mice(train_mean %>% select("ps_Q_Creative", "re_Q_havesibling", sample(nonselect_Q,10)), seed=1234, m=5)
complete_10 <- complete(mice_result_10)
train_4$ps_Q_Creative <- complete_10$ps_Q_Creative
train_4$re_Q_havesibling <- complete_10$re_Q_havesibling
train_4 %>% is.na() %>% colSums()

#fwrite(train_4, file = "train_mean.csv", row.names = F)

# ------------- 새롭게 만든 데이터에 조나단 스코어링 진행 ------- #
train_mean_jo <- train_4
compute_score_mean = function(row,col,type,data) {
  if (!is.na(data[row,col])) {
    if (type == 'careness') {
      if (col %in% c('ps_Q_GoodLiar','ps_Q_Feminist','ps_Q_SupportCharity','ps_Q_ResultOfCircumstance')) {
        if (data[row,col] == 1) return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'fairness') {
      if (col %in% c('mo_Q_minwage_job','edu_Q_parents_college','edu_Q_publicschool','ps_Q_Optimist','ps_Q_GoodLiar','ps_Q_Normal','ps_Q_Feminist','ps_Q_SupportCharity','ps_Q_ResultOfCircumstance')) {
        if (data[row,col] == 1) return(1)
        else return(0)
      }
      else if (col %in% c('ps_Q_BuyHappiness','ps_Q_happy>right')) {
        if (data[row,col] == 0) return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'loyalty') {
      if (col %in% c('ps_Q_LikeFirstName','ps_Q_Obedient','ps_Q_Idealist','env_Q_same_state','ps_Q_RiskFriendly')) {
        if (data[row,col] == 1) return(1)
        else return(0)
      }
      else if (col %in% c('ps_Q_Idealist')) {
        if (data[row,col] == 0) return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'authority') {
      if (col %in% c('re_Q_successfulthan_friend','mo_Q_own_residence','mo_Q_live_metro','mo_Q_has_enoughcash_now','mo_Q_fulltimejob','edu_Q_parents_college','edu_Q_mas/doc_degree','Life_Q_gun','ps_Q_BuyHappiness','ps_Q_Obedient','ps_Q_happy>right','env_Q_p_sup_dem')) {
        if (data[row,col] == 1) return(1)
        else return(0)
      }
      else if (col %in% c('mo_Q_has_debt','mo_Q_feel_financialdebt','mo_Q_poor','mo_Q_carpayment','mo_Q_minwage_job','edu_Q_publicschool','ps_Q_Feminist','ps_Q_Idealist')) {
        if (data[row,col] == 0) return(1)
        else return(0)
      }
      else return(0)
    }
    else if (type == 'sanctity') {
      if (col %in% c('Life_Q_medipray','ps_Q_happy>right','ps_Q_RiskFriendly')) {
        if (data[row,col] == 1) return(1)
        else return(0)
      }
      else if (col %in% c('edu_Q_publicschool')) {
        if (data[row,col] == 1) return(1)
        else return(0)
      }
      else return(0)
    }
    else return(0)
  }
  else return(0)
}

jonathan_mean = function(type, data) {
  new_col = rep(NA,nrow(data))
  cols = colnames(data)[grep('Q_',colnames(data))]
  
  for (i in 1:nrow(data)) {
    score = 0
    
    for (col in cols) {
      score = score + compute_score_mean(i,col,type,data)
    }
    new_col[i] = score
  }
  return(new_col)
}
train_mean_jo$careness = jonathan_mean('careness', train_4)
train_mean_jo$fairness = jonathan_mean('fairness', train_4)
train_mean_jo$loyalty = jonathan_mean('loyalty', train_4)
train_mean_jo$authority = jonathan_mean('authority',train_4)
train_mean_jo$sanctity = jonathan_mean('sanctity', train_4)

# ----------- 새롭게 만든 sum파생변수 적용 ------------ #
train_mean_jo_sum <- train_mean_jo

train_mean_jo_sum$Life_sum = ifelse(train_mean_jo_sum$Life_Q_exerc3we == 1,1,0) +
  ifelse(train_mean_jo_sum$Life_Q_breakfast == 1,1,0) +
  ifelse(train_mean_jo_sum$Life_Q_earlyAlarm == 1,1,0) +
  ifelse(train_mean_jo_sum$Life_Q_standardTime == 1,1,0)

train_mean_jo_sum$edu_sum = ifelse(train_mean_jo_sum$edu_Q_allA == 1,1,0) +
  ifelse(train_mean_jo_sum$edu_Q_goodat_math == 1,1,0) +
  ifelse(train_mean_jo_sum$edu_Q_parents_college == 1,1,0) +
  ifelse(train_mean_jo_sum$edU_Q_mas_doc_degree == 1,1,0)
train_mean_jo_sum %>% glimpse()

fwrite(train_mean_jo_sum, file = "train_mean_jo_sum.csv", row.names = F)
# ------------- DATA2: MICE with NA -------------- #
# train_5는 새롭게 채울 데이터
train_5 <- train_3
train_5[,10:108] %<>% mutate_if(is.character, as.factor)
train_5[,setdiff(feature,c("Income","EducationLevel"))] %<>% mutate_if(is.character, as.factor)
train_5 %>% glimpse()
colSums(is.na(train_5))

# 질문들간의 연관 있는 것들로 NA를 채워보자!
mice_result_1s <- mice(train_5 %>% select("mo_Q_fulltimejob","mo_Q_minwage_job",nonselect_Q[1:10]), seed=1234, m=5)
complete_1s <- complete(mice_result_1s)
train_5$mo_Q_fulltimejob <- complete_1s$mo_Q_fulltimejob
train_5$mo_Q_minwage_job <- complete_1s$mo_Q_minwage_job
for (i in 1:10) {
  train_5[,nonselect_Q[i]] <- complete_1s[,nonselect_Q[i]]
}

mice_result_2s <- mice(train_5 %>% select("Life_Q_collectHobby", "mo_Q_has_enoughcash_now","edu_Q_publicschool",nonselect_Q[11:20]), seed=1234, m=5)
# 3개 연관된거랑 2개 연관된 거 또 다르네 시간이 있으면 둘이 비교해보고 아니면 3개로 넣자
complete_2s <- complete(mice_result_2s)
train_5$Life_Q_collectHobby <- complete_2s$Life_Q_collectHobby
train_5$mo_Q_has_enoughcash_now <- complete_2s$mo_Q_has_enoughcash_now
train_5$edu_Q_publicschool <- complete_2s$edu_Q_publicschool
for (i in 11:20) {
  train_5[,nonselect_Q[i]] <- complete_2s[,nonselect_Q[i]]
}

mice_result_3s <- mice(train_5 %>% select("Life_Q_drink", "re_Q_newromance",nonselect_Q[21:30]), seed=1234, m=5)
complete_3s <- complete(mice_result_3s)
train_5$Life_Q_drink <- complete_3s$Life_Q_drink
train_5$re_Q_newromance <- complete_3s$re_Q_newromance
for (i in 21:30) {
  train_5[,nonselect_Q[i]] <- complete_3s[,nonselect_Q[i]]
}

mice_result_4s <- mice(train_5 %>% select( "Life_Q_gun", "ps_Q_PowerOfPositive",nonselect_Q[31:40]), seed=1234, m=5)
complete_4s <- complete(mice_result_4s)
train_5$Life_Q_gun <- complete_4s$Life_Q_gun
train_5$ps_Q_PowerOfPositive <- complete_4s$ps_Q_PowerOfPositive
for (i in 31:40) {
  train_5[,nonselect_Q[i]] <- complete_4s[,nonselect_Q[i]]
}

mice_result_5s <- mice(train_5 %>% select("edu_Q_parents_college","env_Q_single_parent",nonselect_Q[41:50]), seed=1234, m=5)
complete_5s <- complete(mice_result_5s)
train_5$edu_Q_parents_college <- complete_5s$edu_Q_parents_college
train_5$env_Q_single_parent <- complete_5s$env_Q_single_parent
for (i in 41:50) {
  train_5[,nonselect_Q[i]] <- complete_5s[,nonselect_Q[i]]
}

mice_result_6s <- mice(train_5 %>% select("re_Q_likepeople", "ps_Q_LikeFirstName", nonselect_Q[51:60]), seed=1234, m=5)
complete_6s <- complete(mice_result_6s)
train_5$re_Q_likepeople <- complete_6s$re_Q_likepeople
train_5$ps_Q_LikeFirstName <- complete_6s$ps_Q_LikeFirstName
for (i in 51:60) {
  train_5[,nonselect_Q[i]] <- complete_6s[,nonselect_Q[i]]
}

mice_result_7s <- mice(train_5 %>% select("ps_Q_GoodLiar", "Life_Q_work50h", nonselect_Q[61:70]), seed=1234, m=5)
complete_7s <- complete(mice_result_7s)
train_5$ps_Q_GoodLiar <- complete_7s$ps_Q_GoodLiar
train_5$Life_Q_work50h <- complete_7s$Life_Q_work50h
for (i in 61:70) {
  train_5[,nonselect_Q[i]] <- complete_7s[,nonselect_Q[i]]
}

mice_result_8s <- mice(train_5 %>% select("Life_Q_watchTV","env_Q_p_spank",nonselect_Q[71:78]), seed=1234, m=5)
complete_8s <- complete(mice_result_8s)
train_5$Life_Q_watchTV <- complete_8s$Life_Q_watchTV
train_5$env_Q_p_spank <- complete_8s$env_Q_p_spank
for (i in 71:78) {
  train_5[,nonselect_Q[i]] <- complete_8s[,nonselect_Q[i]]
}

mice_result_9s <- mice(train_5 %>% select("Life_Q_livealone", "ps_Q_LeftHanded", sample(nonselect_Q,10)), seed=1234, m=5)
complete_9s <- complete(mice_result_9s)
train_5$Life_Q_livealone <- complete_9s$Life_Q_livealone
train_5$ps_Q_LeftHanded <- complete_9s$ps_Q_LeftHanded

mice_result_10s <- mice(train_5 %>% select("ps_Q_Creative", "re_Q_havesibling", sample(nonselect_Q,10)), seed=1234, m=5)
complete_10s <- complete(mice_result_10s)
train_5$ps_Q_Creative <- complete_10s$ps_Q_Creative
train_5$re_Q_havesibling <- complete_10s$re_Q_havesibling
train_5 %>% is.na() %>% colSums()

#fwrite(train_5, file = "train_na.csv", row.names = F)
# ------------- 새롭게 만든 데이터에 조나단 스코어링 진행 ------- #
train_na_jo <- train_5
train_na_jo$careness = jonathan('careness', train_5)
train_na_jo$fairness = jonathan('fairness', train_5)
train_na_jo$loyalty = jonathan('loyalty', train_5)
train_na_jo$authority = jonathan('authority',train_5)
train_na_jo$sanctity = jonathan('sanctity', train_5)

# ----------- 새롭게 만든 sum파생변수 적용 ------------ #
train_na_jo_sum <- train_na_jo

train_na_jo_sum$Life_sum = ifelse(train_na_jo_sum$Life_Q_exerc3we == "Yes",1,0) +
  ifelse(train_na_jo_sum$Life_Q_breakfast == "Yes",1,0) +
  ifelse(train_na_jo_sum$Life_Q_earlyAlarm == "Yes",1,0) +
  ifelse(train_na_jo_sum$Life_Q_standardTime == "Yes",1,0)

train_na_jo_sum$edu_sum = ifelse(train_na_jo_sum$edu_Q_allA == "Yes",1,0) +
  ifelse(train_na_jo_sum$edu_Q_goodat_math == "Yes",1,0) +
  ifelse(train_na_jo_sum$edu_Q_parents_college == "Yes",1,0) +
  ifelse(train_na_jo_sum$edU_Q_mas_doc_degree == "Yes",1,0)

train_na_jo_sum %>% glimpse()

fwrite(train_na_jo_sum, file = "train_na_jo_sum.csv", row.names = F)

# --------- DATA3: 모든 변수에 대해 무응답으로 처리! -------------------- #
train_nonanswer <- train_3
for (i in 1:ncol(train_nonanswer)) {
  train_nonanswer[,i] = replace(train_nonanswer[,i], which(is.na(train_nonanswer[,i])), "non_answer")
}

# 모델링 전 데이터 구조 맞추는 코드
train_nonanswer[,c("Income", "EducationLevel", "careness","loyalty", "authority", "sanctity", "fairness")] %<>% mutate_if(is.character, as.integer)
train_nonanswer %<>% mutate_if(is.character, as.factor) 
train_nonanswer %>% glimpse()

#fwrite(train_nonanswer, file = "train_nonanswer.csv", row.names = F)

# ------------- 새롭게 만든 데이터에 조나단 스코어링 진행 ------- #
train_nonanswer_jo <- train_nonanswer
train_nonanswer_jo$careness = jonathan('careness', train_nonanswer)
train_nonanswer_jo$fairness = jonathan('fairness', train_nonanswer)
train_nonanswer_jo$loyalty = jonathan('loyalty', train_nonanswer)
train_nonanswer_jo$authority = jonathan('authority',train_nonanswer)
train_nonanswer_jo$sanctity = jonathan('sanctity', train_nonanswer)

# ----------- 새롭게 만든 sum파생변수 적용 ------------ #
train_nonanswer_jo_sum <- train_nonanswer_jo

train_nonanswer_jo_sum$Life_sum = ifelse(train_nonanswer_jo_sum$Life_Q_exerc3we == "Yes",1,0) +
  ifelse(train_nonanswer_jo_sum$Life_Q_breakfast == "Yes",1,0) +
  ifelse(train_nonanswer_jo_sum$Life_Q_earlyAlarm == "Yes",1,0) +
  ifelse(train_nonanswer_jo_sum$Life_Q_standardTime == "Yes",1,0)

train_nonanswer_jo_sum$edu_sum = ifelse(train_nonanswer_jo_sum$edu_Q_allA == "Yes",1,0) +
  ifelse(train_nonanswer_jo_sum$edu_Q_goodat_math == "Yes",1,0) +
  ifelse(train_nonanswer_jo_sum$edu_Q_parents_college == "Yes",1,0) +
  ifelse(train_nonanswer_jo_sum$edU_Q_mas_doc_degree == "Yes",1,0)

train_nonanswer_jo_sum %>% glimpse()

fwrite(train_nonanswer_jo_sum, file = "train_nonanswer_jo_sum.csv", row.names = F)

# ----------- DATA4: MICE with MCA ---------------------------
train_6 <- train_3
train_6[,10:108] %<>% mutate_if(is.character, as.factor)
train_6 %>% glimpse()
res.impute <- imputeMCA(train_6[,10:108], ncp=5)
# res.mca <- MCA(train_6, tab.disj=res.impute$tab.disj)
MCA_Q_impute <-  res.impute$completeObs
train_mca <- cbind(train_6[,1:9], MCA_Q_impute)
train_mca %>% glimpse()

# ----- imputeMCA + jonathan scoring --------- #
train_mca_jo <- train_mca
train_mca_jo$careness = jonathan('careness', train_mca)
train_mca_jo$fairness = jonathan('fairness', train_mca)
train_mca_jo$loyalty = jonathan('loyalty', train_mca)
train_mca_jo$authority = jonathan('authority',train_mca)
train_mca_jo$sanctity = jonathan('sanctity', train_mca)
train_mca_jo %>% is.na() %>% sum()

# ----------- 새롭게 만든 sum파생변수 적용 ------------ #
train_mca_jo_sum <- train_mca_jo

train_mca_jo_sum$Life_sum = ifelse(train_mca_jo_sum$Life_Q_exerc3we == "Yes",1,0) +
  ifelse(train_mca_jo_sum$Life_Q_breakfast == "Yes",1,0) +
  ifelse(train_mca_jo_sum$Life_Q_earlyAlarm == "Yes",1,0) +
  ifelse(train_mca_jo_sum$Life_Q_standardTime == "Yes",1,0)

train_mca_jo_sum$edu_sum = ifelse(train_mca_jo_sum$edu_Q_allA == "Yes",1,0) +
  ifelse(train_mca_jo_sum$edu_Q_goodat_math == "Yes",1,0) +
  ifelse(train_mca_jo_sum$edu_Q_parents_college == "Yes",1,0) +
  ifelse(train_mca_jo_sum$edU_Q_mas_doc_degree == "Yes",1,0)

train_mca_jo_sum %>% glimpse()

fwrite(train_mca_jo_sum, file = "train_mca_jo_sum.csv", row.names = F)

# ---------- DATA5: MICE with rf ------------------------ #
train_7 <- train_3
train_7[,10:108] %<>% mutate_if(is.character, as.factor)
train_7[,setdiff(feature,c("Income","EducationLevel"))] %<>% mutate_if(is.character, as.factor)
train_7 %>% glimpse()
colSums(is.na(train_7))
train_7 %>% glimpse()

# 질문들간의 연관 있는 것들로 NA를 채워보자!
mice_result_1rf <- mice(train_7 %>% select("mo_Q_fulltimejob","mo_Q_minwage_job",nonselect_Q[1:10]), seed=1234, m=1, method = 'rf')
complete_1rf <- complete(mice_result_1rf)

train_7$mo_Q_fulltimejob <- complete_1rf$mo_Q_fulltimejob
train_7$mo_Q_minwage_job <- complete_1rf$mo_Q_minwage_job
for (i in 1:10) {
  train_7[,nonselect_Q[i]] <- complete_1rf[,nonselect_Q[i]]
}

mice_result_2rf <- mice(train_7 %>% select("Life_Q_collectHobby", "mo_Q_has_enoughcash_now","edu_Q_publicschool",nonselect_Q[11:20]), seed=1234, m=1, method = 'rf')
# 3개 연관된거랑 2개 연관된 거 또 다르네 시간이 있으면 둘이 비교해보고 아니면 3개로 넣자
complete_2rf <- complete(mice_result_2rf)
train_7$Life_Q_collectHobby <- complete_2rf$Life_Q_collectHobby
train_7$mo_Q_has_enoughcash_now <- complete_2rf$mo_Q_has_enoughcash_now
train_7$edu_Q_publicschool <- complete_2rf$edu_Q_publicschool
for (i in 11:20) {
  train_7[,nonselect_Q[i]] <- complete_2rf[,nonselect_Q[i]]
}

mice_result_3rf <- mice(train_7 %>% select("Life_Q_drink", "re_Q_newromance",nonselect_Q[21:30]), seed=1234, m=1, method = 'rf')
complete_3rf <- complete(mice_result_3rf)
train_7$Life_Q_drink <- complete_3rf$Life_Q_drink
train_7$re_Q_newromance <- complete_3rf$re_Q_newromance
for (i in 21:30) {
  train_7[,nonselect_Q[i]] <- complete_3rf[,nonselect_Q[i]]
}

mice_result_4rf <- mice(train_7 %>% select( "Life_Q_gun", "ps_Q_PowerOfPositive",nonselect_Q[31:40]), seed=1234, m=1, method = 'rf')
complete_4rf <- complete(mice_result_4rf)
train_7$Life_Q_gun <- complete_4rf$Life_Q_gun
train_7$ps_Q_PowerOfPositive <- complete_4rf$ps_Q_PowerOfPositive
for (i in 31:40) {
  train_7[,nonselect_Q[i]] <- complete_4rf[,nonselect_Q[i]]
}

mice_result_5rf <- mice(train_7 %>% select("edu_Q_parents_college","env_Q_single_parent",nonselect_Q[41:50]), seed=1234, m=1, method = 'rf')
complete_5rf <- complete(mice_result_5rf)
train_7$edu_Q_parents_college <- complete_5rf$edu_Q_parents_college
train_7$env_Q_single_parent <- complete_5rf$env_Q_single_parent
for (i in 41:50) {
  train_7[,nonselect_Q[i]] <- complete_5rf[,nonselect_Q[i]]
}

mice_result_6rf <- mice(train_7 %>% select("re_Q_likepeople", "ps_Q_LikeFirstName", nonselect_Q[51:60]), seed=1234, m=1, method = 'rf')
complete_6rf <- complete(mice_result_6rf)
train_7$re_Q_likepeople <- complete_6rf$re_Q_likepeople
train_7$ps_Q_LikeFirstName <- complete_6rf$ps_Q_LikeFirstName
for (i in 51:60) {
  train_7[,nonselect_Q[i]] <- complete_6rf[,nonselect_Q[i]]
}

mice_result_7rf <- mice(train_7 %>% select("ps_Q_GoodLiar", "Life_Q_work50h", nonselect_Q[61:70]), seed=1234, m=1, method = 'rf')
complete_7rf <- complete(mice_result_7rf)
train_7$ps_Q_GoodLiar <- complete_7rf$ps_Q_GoodLiar
train_7$Life_Q_work50h <- complete_7rf$Life_Q_work50h
for (i in 61:70) {
  train_7[,nonselect_Q[i]] <- complete_7rf[,nonselect_Q[i]]
}

mice_result_8rf <- mice(train_7 %>% select("Life_Q_watchTV","env_Q_p_spank",nonselect_Q[71:78]), seed=1234, m=1, method = 'rf')
complete_8rf <- complete(mice_result_8rf)
train_7$Life_Q_watchTV <- complete_8rf$Life_Q_watchTV
train_7$env_Q_p_spank <- complete_8rf$env_Q_p_spank
for (i in 71:78) {
  train_7[,nonselect_Q[i]] <- complete_8rf[,ct_Q[i]]
}

mice_result_9rf <- mice(train_7 %>% select("Life_Q_livealone", "ps_Q_LeftHanded", sample(ct_Q,10)), seed=1234, m=1, method = 'rf')
complete_9rf <- complete(mice_result_9rf)
train_7$Life_Q_livealone <- complete_9rf$Life_Q_livealone
train_7$ps_Q_LeftHanded <- complete_9rf$ps_Q_LeftHanded

mice_result_10rf <- mice(train_7 %>% select("ps_Q_Creative", "re_Q_havesibling", sample(nonselect_Q,10)), seed=1234, m=1, method = 'rf')
complete_10rf <- complete(mice_result_10rf)
train_7$ps_Q_Creative <- complete_10rf$ps_Q_Creative
train_7$re_Q_havesibling <- complete_10rf$re_Q_havesibling
train_7 %>% is.na() %>% colSums()

#fwrite(train_7, file = "train_rf.csv", row.names = F)

# ------------- 새롭게 만든 데이터에 조나단 스코어링 진행 ------- #
train_rf_jo <- train_7
train_rf_jo$careness = jonathan('careness', train_7)
train_rf_jo$fairness = jonathan('fairness', train_7)
train_rf_jo$loyalty = jonathan('loyalty', train_7)
train_rf_jo$authority = jonathan('authority',train_7)
train_rf_jo$sanctity = jonathan('sanctity', train_7)

# ----------- 새롭게 만든 sum파생변수 적용 ------------ #
train_rf_jo_sum <- train_rf_jo

train_rf_jo_sum$Life_sum = ifelse(train_rf_jo_sum$Life_Q_exerc3we == "Yes",1,0) +
  ifelse(train_rf_jo_sum$Life_Q_breakfast == "Yes",1,0) +
  ifelse(train_rf_jo_sum$Life_Q_earlyAlarm == "Yes",1,0) +
  ifelse(train_rf_jo_sum$Life_Q_standardTime == "Yes",1,0)

train_rf_jo_sum$edu_sum = ifelse(train_rf_jo_sum$edu_Q_allA == "Yes",1,0) +
  ifelse(train_rf_jo_sum$edu_Q_goodat_math == "Yes",1,0) +
  ifelse(train_rf_jo_sum$edu_Q_parents_college == "Yes",1,0) +
  ifelse(train_rf_jo_sum$edU_Q_mas_doc_degree == "Yes",1,0)

train_rf_jo_sum %>% glimpse()

fwrite(train_rf_jo_sum, file = "train_rf_jo_sum.csv", row.names = F)

## ------ 최종데이터 셋만 남기기 ------------ ##
rm(list = setdiff(ls(),c("train_mean_jo_sum", "train_nonanswer_jo_sum", "train_na_jo_sum", "train_mca_jo_sum", "train_rf_jo_sum")))


# ------> variable selection & test set.R 로 이어집니다.