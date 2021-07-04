####-----------------------------------------------------------###
####-------- M2 APE Machine Learning in Economics -------------###
####-----------------------------------------------------------###

# Predicting Bernie's and Trump's Tweets

rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Packages

list.of.packages <- c("tidyverse", "nnls", 'kableExtra', "quadprog", "SuperLearner", 
                      "ggplot2", "raster", "sp", "rgdal", "rgeos", "glmnet", "Matrix", 
                      "foreach", "KernelKnn", "randomForest", "FactoMineR",'kernlab', 'data.table','httr','zip')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

lapply(list.of.packages, require, character.only=TRUE)

select <- dplyr::select

# Loading datasets

dir = '/Users/usuario/Downloads/'
github_link <- "https://github.com/rpazos-h/ML_Trump_Bernie_tweets/raw/main/files.zip"

temp_file <- paste(dir,'files.zip',sep='')
req <- GET(github_link, 
           authenticate(Sys.getenv("GITHUB_PAT"), ""),
           write_disk(path = temp_file))
unzip(temp_file, exdir = dir,junkpaths=TRUE)

tidy_w2v_test <- readRDS(paste(dir,'test_w2v.rds',sep=""))
tidy_w2v_train <- readRDS(paste(dir,"train_w2v.rds",sep=""))
tidy_tweets_topwords_test <- readRDS(paste(dir,"test_top500_words.rds",sep=""))
tidy_tweets_topwords_train <- readRDS(paste(dir,"train_top500_words.rds",sep=""))
freq_words <- readRDS(paste(dir,"freq_words.rds",sep=""))


# Setting Bernie=1 in author

author <- ifelse(tidy_tweets_topwords_train[,2]=='bernie',1,0)

training_topwords <- tidy_tweets_topwords_train[,3:503]
training_w2v <- tidy_w2v_train[,3:357]

testing_topwords <- tidy_tweets_topwords_test[,2:502]
testing_w2v <- tidy_w2v_test[,2:356]


#------ Section 1. Further cleaning of TOP WORDS -----

freq_words <- freq_words %>% 
  arrange(word) %>% 
  mutate(first_characters = substr(word,1,5))

freq_words_2 <- freq_words %>% 
  group_by(first_characters) %>% 
  tally()

freq_words_3 <- freq_words %>% 
  left_join(freq_words_2, by = 'first_characters') %>% 
  group_by(first_characters) %>% 
  summarise(max = max(n.x)) %>% 
  ungroup() %>% 
  mutate(filter = 1)

freq_words<-freq_words %>% 
  left_join(freq_words_3, by = c('first_characters','n'='max')) %>% 
  mutate(filter = ifelse(is.na(filter),0,filter)) %>% 
  filter(filter==0) %>% 
  select(word)

training_topwords <- training_topwords %>% 
  select(-(freq_words$word), -"'s", -"'t")

testing_topwords <- testing_topwords %>% 
  select(-(freq_words$word), -"'s", -"'t")


#---- Dimensionality reduction -----

### training set 
pcares <- PCA(training_w2v, ncp = 30, graph = FALSE)
training_w2v_pca <- pcares$ind$coord

### testing set
pcapred <- predict.PCA(pcares, testing_w2v)
testing_w2v_pca <- pcapred$coord

## Training set 
pcares_w <- PCA(training_topwords, ncp = 30, graph = FALSE)
training_topwords_pca <- pcares_w$ind$coord

## Test set
pcapred_w <- predict.PCA(pcares_w, testing_topwords)
testing_topwords_pca <- pcapred_w$coord

#---- Section 2. Models

#   Linear Model ----

# All features

sl_lm = SuperLearner(Y = author,
                     X = data.frame(training_w2v), 
                     family = binomial(), 
                     SL.library = "SL.lm", 
                     cvControl = list(V=0))

sl_lm_w = SuperLearner(Y = author, 
                       X = data.frame(training_topwords), 
                       family = binomial(), 
                       SL.library = "SL.lm", 
                       cvControl = list(V=0))
# PCA 

sl_lm_pca = SuperLearner(Y = author, 
                         X = data.frame(training_w2v_pca), 
                         family = binomial(), 
                         SL.library = "SL.lm", 
                         cvControl = list(V=0))

sl_lm_pca_w = SuperLearner(Y = author, 
                           X = data.frame(training_topwords_pca), 
                           family = binomial(), 
                           SL.library = "SL.lm", 
                           cvControl = list(V=0))

# -----GLM ------

sl_glm_pca = SuperLearner(Y = author, 
                          X = data.frame(training_w2v_pca), 
                          family = binomial(), 
                          SL.library = "SL.glm", 
                          cvControl = list(V=0))

sl_glm_w_pca = SuperLearner(Y = author, 
                            X = data.frame(training_topwords_pca), 
                            family = binomial(), 
                            SL.library = "SL.glm", 
                            cvControl = list(V=0))

# Kernel nearest neighbor -

sl_k_pca = SuperLearner(Y = author, 
                        X = data.frame(training_w2v_pca), 
                        family = binomial(), 
                        SL.library = "SL.kernelKnn", 
                        cvControl = list(V=0))

sl_k_w_pca = SuperLearner(Y = author, 
                          X = data.frame(training_topwords_pca), 
                          family = binomial(), 
                          SL.library = "SL.kernelKnn", 
                          cvControl = list(V=0))

#--- Support Vector Machine -----

sl_svm_pca = SuperLearner(Y = author, 
                          X = data.frame(training_w2v_pca), 
                          family = binomial(), 
                          SL.library = "SL.ksvm", 
                          cvControl = list(V=0))
sl_svm_w_pca = SuperLearner(Y = author, 
                            X = data.frame(training_topwords_pca), 
                            family = binomial(), 
                            SL.library = "SL.ksvm", 
                            cvControl = list(V=0))

#-  Elastic Net Regularisation ----

ridge = create.Learner("SL.glmnet", params = list(alpha = 0), name_prefix="ridge")
lasso = create.Learner("SL.glmnet", params = list(alpha = 1), name_prefix="lasso")

## Lasso
sl_lasso_pca = SuperLearner(Y = author, 
                            X = data.frame(training_w2v_pca), 
                            family = binomial(), 
                            SL.library = lasso$names, 
                            cvControl = list(V=0))

# Coefficients of lasso
coef(sl_lasso_pca$fitLibrary$lasso_1_All$object)


## Ridge
sl_ridge_pca = SuperLearner(Y = author, 
                            X = data.frame(training_w2v_pca), 
                            family = binomial(), 
                            SL.library = ridge$names, 
                            cvControl = list(V=0))
#

sl_lasso_w_pca = SuperLearner(Y = author, 
                              X = data.frame(training_topwords_pca), 
                              family = binomial(), 
                              SL.library = lasso$names, 
                              cvControl = list(V=0))
# Coefficients of lasso
coef(sl_lasso_w_pca$fitLibrary$lasso_1_All$object)


## Ridge
sl_ridge_w_pca = SuperLearner(Y = author, 
                              X = data.frame(training_topwords_pca), 
                              family = binomial(), 
                              SL.library = ridge$names, 
                              cvControl = list(V=0))

# -- Random Forest ----

sl_rf_pca = SuperLearner(Y = author, 
                         X = data.frame(training_w2v_pca), 
                         family = binomial(), 
                         SL.library = "SL.randomForest", cvControl = list(V=0))
sl_rf_w_pca = SuperLearner(Y = author, 
                           X = data.frame(training_topwords_pca), 
                           family = binomial(), SL.library = "SL.randomForest", cvControl = list(V=0))

#-  Ensemble Models -----

## Ridge and Lasso
Ridge_Lasso <- c(lasso$names, ridge$names)

sl_en1_pca <- SuperLearner(Y = author,
                           X = data.frame(training_w2v_pca), 
                           family = binomial(),
                           SL.library = Ridge_Lasso, 
                           cvControl = list(V=0))

# Get coefficients of lasso
coef(sl_en1_pca$fitLibrary$lasso_1_All$object)

# Overall performance
cv_rl = CV.SuperLearner(Y = author, 
                        X = data.frame(training_w2v_pca), 
                        family = binomial(), 
                        SL.library = Ridge_Lasso, 
                        cvControl = list(V=5))
summary(cv_rl)


### All models
sl_en_pca <- SuperLearner(Y = author,
                          X = data.frame(training_w2v_pca), family = binomial(),
                          SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, 
                                         "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), cvControl = list(V=0))
sl_en_pca

cv_all = CV.SuperLearner(Y = author,
                         X = data.frame(training_w2v_pca), family = binomial(), 
                         SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), 
                         cvControl = list(V=5))
summary(cv_all)


### TW

## Ridge and Lasso
sl_en1_w_pca <- SuperLearner(Y = author,
                             X = data.frame(training_topwords_pca), family = binomial(),
                             SL.library = Ridge_Lasso, cvControl = list(V=0))
sl_en1_w_pca

coef(sl_en1_w_pca$fitLibrary$lasso_1_All$object)

cv_rl_w = CV.SuperLearner(Y = author, 
                          X = data.frame(training_topwords_pca), 
                          family = binomial(), SL.library = Ridge_Lasso, cvControl = list(V=5))
summary(cv_rl_w)


### All models
sl_en_w_pca <- SuperLearner(Y = author,
                            X = data.frame(training_topwords_pca), family = binomial(),
                            SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), cvControl = list(V=0))
sl_en_w_pca

cv_all_w = CV.SuperLearner(Y = author, 
                           X = data.frame(training_topwords_pca), 
                           family = binomial(), 
                           SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), 
                           cvControl = list(V=5))
summary(cv_all_w)


#--- Models training Emb + TW together -----

#------ Ensemble Methods 1 ----

#### PCA data

## Ridge and Lasso  - just to compare to above
sl_en1_pca_2 <- SuperLearner(Y = author,
                             X = data.frame(training_w2v_pca, training_topwords_pca), family = binomial(),
                             SL.library = Ridge_Lasso, cvControl = list(V=0))
sl_en1_pca_2

coef(sl_en1_pca_2$fitLibrary$lasso_1_All$object)

cv_rl_pca = CV.SuperLearner(Y = author, 
                            X = data.frame(training_w2v_pca, training_topwords_pca), 
                            family = binomial(), 
                            SL.library = Ridge_Lasso, cvControl = list(V=5))
summary(cv_rl_pca)


### All 
sl_en_pca_2 <- SuperLearner(Y = author,
                            X = data.frame(training_w2v_pca, training_topwords_pca), family = binomial(),
                            SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), cvControl = list(V=0))
sl_en_pca_2

cv_all_pca = CV.SuperLearner(Y = author, 
                             X = data.frame(training_w2v_pca, training_topwords_pca), family = binomial(), SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), 
                             cvControl = list(V=2))
summary(cv_all_pca)

#### Screening for both (using Lasso)

SL.library <- list(c("SL.lm", "screen.glmnet"), c("SL.glm", "screen.glmnet"), lasso$names, ridge$names, c("SL.randomForest", "screen.glmnet"), c("SL.kernelKnn", "screen.glmnet"), c("SL.ksvm", "screen.glmnet"))
sl_screen <- SuperLearner(Y = author,
                          X = data.frame(training_w2v, training_topwords), family = binomial(),
                          SL.library = SL.library, cvControl = list(V=0))
sl_screen

coef(sl_screen$fitLibrary$lasso_1_All$object)

SL.library <- list(c("SL.lm", "screen.glmnet"), c("SL.glm", "screen.glmnet"), lasso$names, ridge$names, c("SL.randomForest", "screen.glmnet"), c("SL.kernelKnn", "screen.glmnet"), c("SL.ksvm", "screen.glmnet"))
sl_screen2 <- SuperLearner(Y = author,
                           X = data.frame(training_topwords_pca, training_w2v_pca), family = binomial(),
                           SL.library = SL.library, cvControl = list(V=0))
sl_screen2

coef(sl_screen$fitLibrary$lasso_1_All$object)


# --------Table with results-----------

list_obj <- ls(pattern = "sl_")

df <- setDT(as.data.frame(get(list_obj[1])$cvRisk), keep.rownames=TRUE)

for (i in list_obj[2:24]){
  mod <- paste("mod",i, sep = "_")
  assign(mod, setDT(as.data.frame(get(i)$cvRisk), keep.rownames=TRUE))
}

list_mod <- ls(pattern = "mod_")

for (i in list_mod){
  df = left_join(df,as.data.frame(get(i)),by='rn')
}
names(df)<-c('model','mod_sl_en_pca',list_mod)

kable(df[,c(1:7,22,23)], digits = 3, format='latex', caption = "MSE - Results Meanings Database", align = 'c') 

#Plots
barplot <- colSums(df[,c(8,10,12,15,18,20,24)], na.rm = TRUE)
barplot2 <- colSums(df[,c(9,11,13,16,19,21,25)], na.rm = TRUE)

par(las=2)
barplot(barplot, horiz=TRUE, legend = rownames(barplot), 
        names.arg = c('glm', 'knn', 'lasso', 'lm', 'rf', 'ridge', 'svm'),
        xlab='MSE', col=rgb(0.2,0.4,0.6,0.6),
        xlim = c(0,0.16), main='Embeddings dataset')

par(las=2)
barplot(barplot2, horiz=TRUE, legend = rownames(barplot), 
        names.arg = c('glm', 'knn', 'lasso', 'lm', 'rf', 'ridge', 'svm'),
        xlab='MSE', col=rgb(0.2,0.4,0.6,0.6),
        xlim = c(0,0.16), main='Top-words dataset')

rm(list= c(ls(pattern = "mod_"),ls(pattern = "sl_")))


#------ Test set with labels -------

training_labeled <- as.data.frame(cbind(author,training_w2v_pca, training_topwords_pca))
names(training_labeled)[2:61] <- paste("Dim", c(1:60),sep=".")

## Size 
smp_size <- floor(0.80 * nrow(training_labeled))
set.seed(36)
ind <- sample(seq_len(nrow(training_labeled)), size = smp_size)

training_wlabels <- training_labeled[ind, ]
testing_wlabels <- training_labeled[-ind, ]

author_train <- training_wlabels$author
author_test <- testing_wlabels$author

training_wlabels <- training_wlabels %>% select(-author)
testing_wlabels <- testing_wlabels %>% select(-author)

#### Fitting on training subset

sl_en_2_test <- SuperLearner(Y = author_train,
                             X = data.frame(training_wlabels), family = binomial(),
                             SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), cvControl = list(V=0))

sl_en_2_test

#### Predicting

## Using the sl_en_test model
pred_sl_en_2_test <- predict(sl_en_2_test, data.frame(testing_wlabels), onlySL = T)
pred_sl_en_2_test

## Check performance

performance <- as.data.frame(cbind(pred_sl_en_2_test$pred, author_test))
performance$V1 <- ifelse(performance$V1 > 0.5, 1, 0)
performance$Correct <- ifelse(performance$V1 == performance$author_test, 1, 0)
table(performance$Correct)
mean(performance$Correct) #0.90

#-- Over the original test set ####

sl_en_pca_2 <- SuperLearner(Y = author,
                            X = data.frame(training_w2v_pca, training_topwords_pca), family = binomial(),
                            SL.library = c("SL.lm", "SL.glm", lasso$names, ridge$names, "SL.randomForest", "SL.kernelKnn", "SL.ksvm"), cvControl = list(V=0))

pred_sl_en_pca_2_test <- predict(sl_en_pca_2, data.frame(testing_w2v_pca, testing_topwords_pca), onlySL = T)
pred_sl_en_pca_2_test

prediction <-  as.data.frame(pred_sl_en_pca_2_test)
prediction$pred <- ifelse(prediction$pred>0.5, 1, 0)

table(prediction$pred)

write.table(prediction$pred,"Rodrigo_Pazos_prediction.txt", sep="",col.names = FALSE, row.names = FALSE)
