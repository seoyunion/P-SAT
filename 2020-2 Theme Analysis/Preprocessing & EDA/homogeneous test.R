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

## ------ Data load ------ ##
train_EDA <- fread(file = "train_EDA.csv", header = T, stringsAsFactors = F, data.table = F)
train_rf_jo_sum_final <- fread(file = "./modeling data/train_rf_jo_sum_final.csv", header = T, stringsAsFactors = F, data.table = F)


## ------- 데이터 동질성 검정 및 plot 제작 --------- ##
# feminist 질문
raw_fem <- train_EDA %>% group_by(ps_Q_Feminist , Party) %>% na.omit() %>% dplyr::summarise(count = n())
raw_spread <- spread(data = raw_fem, key = Party, value = count)
rf_fem <- train_rf_jo_sum_final %>% group_by(ps_Q_Feminist, Party) %>% dplyr::summarise(count = n())
rf_spread <- spread(data = rf_fem, key = Party, value = count)
homogeneous_table <- rbind(raw_spread[2,2:3], rf_spread[2,2:3]) %>% as.matrix()
rownames(homogeneous_table) <- c("raw_fem_yes","rf_fem_yes")
chisq.test(homogeneous_table)
