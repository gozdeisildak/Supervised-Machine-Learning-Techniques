##Ferhat KORTAK 
##GÖZDE ISILDAK
##SERAY ESEN 

library(class)
library(ISLR)
library(caret)
library(dummies)
library(e1071)
library(purrr)
library(tidyr)
library(ggplot2)
library(tree) 
library(neuralnet)
library(party)
library(randomForest)
library(Hmisc)
library(ROCR)

online_shopper <- read.table("D:/machine_learning/online_shoppers_intention.csv", header = TRUE , sep =",", dec =".")
#----------------DATA CLEANING---------------------------

#Remove Duplicatess

isDuplicated <- duplicated(online_shopper)  # Defines same instance.If there is sameinstance , It be True.
duplicatedValues <- online_shopper[isDuplicated, ] #Same instance' values
num_duplicatedRow <- nrow(duplicatedValues) #Number of same instances
duplicated_online_shopper <- online_shopper[!duplicated(online_shopper), ] #Diffrent instance' values  
num_datasetRow <- nrow(duplicated_online_shopper) #Number of distinct instances

#Missing Values (mean/median for feature)  
is.na(duplicated_online_shopper)      
apply(is.na(duplicated_online_shopper),2,sum)
which (is.na(duplicated_online_shopper)) # Which one is NA?   
duplicated_online_shopper[!complete.cases(duplicated_online_shopper),] #The function complete.cases() returns a logical vector indicating which cases are complete.    *

#Dummy attribute 
#Convert categorical feature to dummy attribute(0/1).
dummy_onlineShoppers<- dummy.data.frame(duplicated_online_shopper, names = c("Month","VisitorType","Weekend") ,omit.constants=FALSE,dummy.classes = getOption("dummy.classes"))
num_descriptiveFeature <- ncol(dummy_onlineShoppers)
num_instance <- nrow(dummy_onlineShoppers)

#NORMALIZATION 
#Range
online_shopper_range_normalization <- as.data.frame(apply(dummy_onlineShoppers[, 1:num_descriptiveFeature], 2, function(x) (x - min(x))/(max(x)-min(x)))) 
online_shopper_range_normalization[, 'Revenue'] <- as.factor(dummy_onlineShoppers[, 'Revenue'])
online_shopper_range_normalization
prepared_onlineShopper <- online_shopper_range_normalization

#----------------CORRELATION MATRIX----
# 1) Correlation Mat$ix
onlineShopper_Con<- dummy.data.frame(prepared_onlineShopper, names = c("Revenue") ,omit.constants=FALSE,dummy.classes = getOption("dummy.classes"))

co_matrix <- cor(onlineShopper_Con)
round(co_matrix, 2)

# 2) correlation find p and n

#The output of the function rcorr() is a list containing the following elements : - r : the correlation matrix - 
# n : the matrix of the number of observations used in analyzing each pair of variables 
#- P : the p-values corresponding to the significance levels of correlations.
# Extract the correlation coefficients

rcorr_matrix <- rcorr(as.matrix(onlineShopper_Con),type = c("pearson","spearman"))
rcorr_matrix

rcorr_matrix$r
round(rcorr_matrix$r, 2)

# Extract p-values
rcorr_matrix$P
round(rcorr_matrix$P, 2)


# 3) flattenCorrMatrix

# ++++++++++++++++++++++++++++
# flattenCorrMatrix
# ++++++++++++++++++++++++++++
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

res2<-rcorr(as.matrix(onlineShopper_Con))
flattenCorrMatrix(res2$r, res2$P)

# 4 ) correlogram

library(corrplot)
corrplot(co_matrix, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

# 5 ) heatmap
# Get some colors

col<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = co_matrix, col = col, symm = TRUE)


#Displays distribution of Target Feature ------                
countsTarget <- table(prepared_onlineShopper$Revenue) #Creates table of distribution
barplot(countsTarget,ylim = c(0,12000),main = "Target Feature Distribution",xlab = "Target Feature")  #Creates bar plot of distribution

#------------------------------CHOOSE MODEL-------------------------------
dt_acc <- numeric()
set.seed(1815850)
#Creates matrix for each evaluation metric.
matrixAcc <-matrix(0,7,5) 
matrixPre <-matrix(0,7,5)
matrixRecall <-matrix(0,7,5)
matrixF1score <-matrix(0,7,5)
matrixAUC <-matrix(0,7,5)
#When cross validaiton done , prints average of evaluation metrics.
printMetricResult <- function(i){
  cat("mean accuracy: ", mean(matrixAcc[i,]),"\n")
  cat("mean precision: ", mean(matrixPre[i,]),"\n")
  cat("mean recall: ", mean(matrixRecall[i,]),"\n")
  cat("mean Fscore: ", mean(matrixF1score[i,]),"\n")
  cat("mean AUC: ", mean(matrixF1score[i,]),"\n")
  
}
#When cross validaiton done , prints names of models.
nameModel <- function(k){
  if(k == 1)  print(">> Knn MODEL --")
  else if(k == 2)  print(">> Naive-Bayes MODEL --")
  else if(k == 3)  print(">> Support Vectör LINEAR MODEL --")
  else if(k == 4)  print(">> Support Vectör POLYNOMIAL MODEL --")
  else if(k == 5)  print(">> Support Vectör RADIAL MODEL --")
  else if(k == 6)  print(">> Desýcýon TREE --")
  else   print(">> Random Forest MODEL --")
}
#AUC FUNCTION--------------
fun.auc <- function(pred,obs){
  # Run the ROCR functions for AUC calculation
  ROC_auc <- performance(prediction(pred,obs),"auc")
  # AUC value
  AUC <- ROC_auc@y.values[[1]] # AUC
  return(AUC)
}

# # ----------------------UNDERSAMPLING--------------------------------------------
online_shopper_df <- prepared_onlineShopper
for(t in 1:2){
  prepared_onlineShopper <- online_shopper_df
  true_index <- which(prepared_onlineShopper$Revenue==TRUE)
  all_false_indexes <- which(prepared_onlineShopper$Revenue==FALSE)
  false_index <- sample (c(1:length(all_false_indexes)), size= length(true_index), replace=F)
  # Check sizes of true and false portion whether equal or not
  length(false_index)
  length(true_index)
  length(all_false_indexes)
  # Merge True and False Portions
  selected_indexes <- cbind(true_index, false_index)
  length(selected_indexes)
  # Filter main dataframe with selected T and F indexes
  prepared_onlineShopper <- prepared_onlineShopper[selected_indexes,]
  dim(prepared_onlineShopper)

for(fold in 1:5){ #Cross Validation 
  cat(paste(fold,". Fold Cross Validation :", "\n",sep=" "))
  print(">> Training and Testing is doing !")
  online_shopper_sample <- sample(1:nrow(prepared_onlineShopper), size=nrow(prepared_onlineShopper)*0.7) #Data sampling for %70
  online_shopper_training <- prepared_onlineShopper[ online_shopper_sample,] #data training
  online_shopper_testing <- prepared_onlineShopper[-online_shopper_sample,] #data testing
  num_train <- ncol(online_shopper_training) #number of coloumn of training set
  num_test <- nrow(online_shopper_testing)#number of row of training set
  
  print(">> Training and Testing is done !")
  
  ##KNN model----------------------------------------------------
  print(">> KNN MODEL << ")
  #First try to determine the right K-value 
  online_shopper_acc <- numeric() #holding variable
  online_shopper_pre <- numeric() #holding variable
  online_shopper_recall <- numeric() #holding variable
  online_shopper_Fscore <- numeric()  #holding variable
  online_shopper_auc <- numeric() #holding variable
  maxAccuracy <- -1
  maxPrecision <- -1
  maxRecall <- -1
  maxFscore <- -1
  maxAUC <- -1
  ValueK <- matrix(0,5,1) # for maximum metric' values k values.
 
  for(i in 1:100){ 
    #Apply knn with k = i
    #Calculates run time
    start_time <- Sys.time()
    knn_Model <- knn(train=online_shopper_training[,-num_train], test=online_shopper_testing[,-num_train], cl=online_shopper_training$Revenue, k=i)
    
    end_time <- Sys.time()
    total_time <- end_time - start_time
    print(total_time)
    #Creates confussion matrix and other metrics
    Knn_Confussion_Matrix <- table(knn_Model,online_shopper_testing$Revenue)
    online_shopper_acc <- c(online_shopper_acc, mean(knn_Model==online_shopper_testing$Revenue))
    online_shopper_pre <-  c(online_shopper_pre,Knn_Confussion_Matrix[1,1]/sum(Knn_Confussion_Matrix[,1]))
    online_shopper_recall <- c(online_shopper_recall,Knn_Confussion_Matrix[1,1]/sum(Knn_Confussion_Matrix[1,]))
    online_shopper_Fscore <- c(online_shopper_Fscore,(2 * (online_shopper_pre * online_shopper_recall) / (online_shopper_pre + online_shopper_recall)))
    online_shopper_auc<-c(online_shopper_auc,fun.auc(ifelse(knn_Model=="TRUE",1,0),as.factor(as.logical(online_shopper_testing$Revenue))))
     #Finds k value for maximum metrics
    if(online_shopper_acc[i] > maxAccuracy){   
      maxAccuracy <- online_shopper_acc[i]
      ValueK[1] <- i
    } 
    if(online_shopper_pre[i] > maxPrecision){  
      maxPrecision <- online_shopper_pre[i]
      ValueK[2] <- i
    }
    if(online_shopper_recall[i] > maxRecall){ 
      maxRecall<- online_shopper_recall[i]
     ValueK[3] <- i
    } 
    if(online_shopper_Fscore[i] > maxFscore){ 
      maxFscore <- online_shopper_Fscore[i]
      ValueK[4] <- i
    } 
    if(online_shopper_auc[i] > maxAUC){ 
      maxAUC <- online_shopper_auc[i]
      ValueK[5] <- i
    } 
  }
  
  #Diffrent Metrics
  
  # 1) Precision Accuracy / Error Rate
  cat(paste("knn_Accuracy:\t", format(mean(online_shopper_acc), digits=4), "\n",sep=" "))
  cat(paste("knn _ Error Rate:\t", format(1-mean(online_shopper_acc), digits=4), "\n",sep=" "))
  
  #Plot error rates for k=1 to 100
  plot(1-online_shopper_acc, type="l", ylab="Error Rate",  xlab="k", main="Error Rate for online_shopper with varying K")
  #Plot accuracy for k=1 to 100
  plot(online_shopper_acc, type="l",col = "red", ylab="Accuracy",  xlab="k", main="Accuracy for online_shopper with varying K")
  matrixAcc[1,fold] <- mean(online_shopper_acc)
 
   # # 2)Confussion Matrix
  Knn_Confussion_Matrix <- table(knn_Model, online_shopper_testing$Revenue)
  barplot(Knn_Confussion_Matrix[ValueK[1][1]] ,ylim = c(0,4000))

  # 3) Precision / Recall
  cat(paste("knn_Precision:\t", format(mean(online_shopper_pre), digits=4), "\n",sep=" "))
  #Plot precision for k=1 to 100
  plot(online_shopper_pre, type="l",col = "red", ylab="PRECISION",  xlab="k", main="Precision for online_shopper with varying K")
  matrixPre[1,fold] <- mean(online_shopper_pre)
  
  cat(paste("knn_Recall:\t", format(mean(online_shopper_recall), digits=4), "\n",sep=" "))
  #Plot recall for k=1 to 100
  plot(online_shopper_recall, type="l",col = "red", ylab="RECALL",  xlab="k", main="Recall for online_shopper with varying K")
  matrixRecall[1,fold] <- mean(online_shopper_recall)
  # 4) f1: score
  cat(paste("knn_F-measure:\t", format(mean(online_shopper_Fscore), digits=4), "\n",sep=" "))
  #Plot fscore for k=1 to 100
  plot(online_shopper_Fscore, type="l",col = "red", ylab="F-MEASURE",  xlab="k", main="F-score for online_shopper with varying K")
  matrixF1score[1,fold] <- mean(online_shopper_Fscore)
  #5) AUC 
  cat(paste("knn_AUC:\t", format(mean(online_shopper_auc), digits=4), "\n",sep=" "))
  matrixAUC[1,fold] <- mean(online_shopper_auc)
 #Draws bar plot with averages of metrics.
  KNN_metricsVal<- c( matrixAcc[1,fold],matrixPre[1,fold],matrixRecall[1,fold],matrixF1score[1,fold], matrixAUC[1,fold])
  barplot(KNN_metricsVal,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Accuracy","Precision","Recall","f1_score","AUC"),col=blues9)
  

  ##NAIVE-BAYES -> Probabilistic----------------------------------------------------------
  
  print(">> NAIVE-BAYES MODEL << ")
  #Calculates run time
  start_time <- Sys.time()
  
  online_shopper_naiveBayes <- naiveBayes(Revenue ~ . ,data = online_shopper_training)
  
  end_time <- Sys.time()
  total_time <- end_time - start_time
  print(total_time)  
  
  #Diffrent Metrics
  #Draws plot and prints to screen
  # 1)Confussion Matrix
  NaiveBayes_prediction <- predict(online_shopper_naiveBayes,  prepared_onlineShopper[-online_shopper_sample, ], type = "class")
  naiveBayes_Confussion_Matrix<-table(NaiveBayes_prediction, prepared_onlineShopper[-online_shopper_sample, "Revenue"],dnn=c("Prediction","Actual"))
  # Plot the Confusion Matrix
  barplot(naiveBayes_Confussion_Matrix ,ylim = c(0,4000))
 
  # 2)  Accuracy / Error Rate
  NaiveBayes_Accuracy <- (sum(naiveBayes_Confussion_Matrix[1,1])+sum(naiveBayes_Confussion_Matrix[2,2]))/length(online_shopper_testing$Revenue)
  cat(paste("naive-Bayes _ Accuracy:\t", format(NaiveBayes_Accuracy, digits=4), "\n",sep=" "))
  cat(paste("naive-Bayes _ Error Rate:\t", format(1-NaiveBayes_Accuracy, digits=4), "\n",sep=" "))
  matrixAcc[2,fold] <- NaiveBayes_Accuracy
  
  # 3) Precision / Recall
  naiveBayes_precision <-  naiveBayes_Confussion_Matrix[1,1]/sum(naiveBayes_Confussion_Matrix[,1])
  cat(paste("naive-Bayes_Precision:\t", format(naiveBayes_precision, digits=4), "\n",sep=" "))
  #Plot precision for k=1 to 100
  matrixPre[2,fold] <- naiveBayes_precision
  
  naiveBayes_recall <- naiveBayes_Confussion_Matrix[1,1]/sum(naiveBayes_Confussion_Matrix[1,])
  cat(paste("naive-Bayes _ Recall:\t", format(naiveBayes_recall, digits=4), "\n",sep=" "))
  matrixRecall[2,fold] <- naiveBayes_recall
  
  # 4) f1: score
  naiveBayes_f1_score <- 2 * (naiveBayes_precision * naiveBayes_recall) / (naiveBayes_precision + naiveBayes_recall)
  cat(paste("Naive-Bayes _ f1 score:\t", format(naiveBayes_f1_score, digits=2), "\n",sep=" "))
  matrixF1score[2,fold] <- naiveBayes_f1_score
  #5) AUC 
  matrixAUC[2,fold] <- fun.auc(ifelse(NaiveBayes_prediction=="TRUE",1,0),ifelse(online_shopper_testing$Revenue=="TRUE",1,0))
  cat(paste("Naive-Bayes _ Auc:\t", format(matrixAUC[2,fold], digits=2), "\n",sep=" "))
  
  #Draws bar plot with averages of metrics.
  NaiveBayes_metricsVal<- c( matrixAcc[2,fold],matrixPre[2,fold],matrixRecall[2,fold],matrixF1score[2,fold], matrixAUC[2,fold])
  barplot(NaiveBayes_metricsVal,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Accuracy","Pre","Recall","fS","AUC"),col=blues9)
  
  # SUPPORT VECTOR MACHINE LINEAR KERNEL-------
  set.seed(50600)
  
  # the reason for using TUNE FUNCTION
  # This generic function tunes hyperparameters of statistical methods using a grid search over supplied parameter ranges.
  # Optimum variables is seen thanks to this algorithm.
  #This algorithm found optimum cost as 4.
  # This algorithm is runned once.It is not necessarily to run again.
  # tuned_parameters <- tune.svm(Revenue~., data = online_shopper_training,kernel="linear" , cost = 2^(-5:5))
  # summary(tuned_parameters )
  
  start_time <- Sys.time()
  
  svmfit <- svm(Revenue ~ ., data = online_shopper_training, kernel = "linear", type="C-classification",cost=4)
  
  end_time <- Sys.time()
  total_time <- end_time - start_time
  print(total_time)  
  
  
  ygrid <- predict(svmfit, online_shopper_testing)
  test_predict <- table(ygrid, online_shopper_testing[, "Revenue"])
  
  #it gives names to table
  rownames(test_predict) <- paste("Actual", rownames(test_predict), sep = ":")
  colnames(test_predict) <- paste("Predicted", colnames(test_predict), sep = ":")
  print(test_predict)
  
  #It is stored to values of all metrics in the matrix
  matrixAcc[3,fold] <-  sum(diag(test_predict)) / sum(test_predict)
  matrixPre[3,fold] <-  test_predict[1,1]/sum(test_predict[,1])
  matrixRecall[3,fold] <- test_predict[1,1]/sum(test_predict[1,])
  matrixF1score[3,fold] <- 2 * ( matrixAcc[3,fold] * matrixRecall[3,fold]) / ( matrixPre[3,fold]+matrixRecall[3,fold])
  matrixAUC[3,fold] <- fun.auc(ifelse(ygrid=="TRUE",1,0),as.factor(as.logical(online_shopper_testing$Revenue)))
  
  svmLinear_metricsVal<- c( matrixAcc[3,fold],matrixPre[3,fold],matrixRecall[3,fold],matrixF1score[3,fold], matrixAUC[3,fold])
  barplot(svmLinear_metricsVal,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Acc","Pre","Recall","fScore","AUC"),col=blues9)
  
  # SUPPORT VECTOR MACHINE POLYNOMIAL KERNEL---------
  
  # the reason for using TUNE FUNCTION
  # This generic function tunes hyperparameters of statistical methods using a grid search over supplied parameter ranges.
  # Optimum variables is seen thanks to this algorithm.
  #This algorithm found optimum gamma as 0.03125 and cost as 8.
  # This algorithm is runned once.It is not necessarily to run again.
  # tuned_parameters <- tune.svm(Revenue~., data = online_shopper_training,kernel="polynomial" ,
  #                              gamma = 2^(-5:-5), cost = 2^(-5:5))  
  # summary(tuned_parameters)

  
  set.seed(50600)
  
  start_time <- Sys.time()
  
  #Degree can be 1:3.
  svmfit = svm(Revenue ~ ., data = online_shopper_training, kernel = "polynomial", gamma=0.03125,cost=8,degree=1)
  
  end_time <- Sys.time()
  total_time <- end_time - start_time
  print(total_time)
  
  ygrid = predict(svmfit, online_shopper_testing)
  test_predict <- table(ygrid, online_shopper_testing[, "Revenue"])
  
  #it gives names to table
  #rownames(test_predict) <- paste("Actual", rownames(test_predict), sep = ":")
  #colnames(test_predict) <- paste("Predicted", colnames(test_predict), sep = ":")
  #print(test_predict)
  
  #It is stored to values of all metrics in the matrix
  matrixAcc[4,fold] <-  sum(diag(test_predict)) / sum(test_predict)
  matrixPre[4,fold] <-  test_predict[1,1]/sum(test_predict[,1])
  matrixRecall[4,fold] <- test_predict[1,1]/sum(test_predict[1,])
  matrixF1score[4,fold] <- 2 * (matrixPre[4,fold] * matrixRecall[4,fold]) / (matrixPre[4,fold]+matrixRecall[4,fold])
  matrixAUC[4,fold] <- fun.auc(ifelse(ygrid=="TRUE",1,0),as.factor(as.logical(online_shopper_testing$Revenue)))
 
  svmPolinomial_metricsVal<- c( matrixAcc[4,fold],matrixPre[4,fold],matrixRecall[4,fold],matrixF1score[4,fold],matrixAUC[4,fold])
  barplot(svmPolinomial_metricsVal,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Acc","Pre","Recall","fscore","AUC"),col=blues9)
  
  # SUPPORT VECTOR MACHINE Radýal KERNEL------

  
  set.seed(50600)
  
  
  # the reason for using TUNE FUNCTION
  # This generic function tunes hyperparameters of statistical methods using a grid search over supplied parameter ranges.
  # Optimum variables is seen thanks to this algorithm.
  #This algorithm found optimum gamma as 0.03125 and cost as 2.
  # This algorithm is runned once.It is not necessarily to run again.
  #tuned_parameters <- tune.svm(Revenue~., data = online_shopper_training,kernel="radial" ,gamma = 2^(-5:-5), cost = 2^(-5:5))
  #summary(tuned_parameters)
  
    start_time <- Sys.time()
    
    svmfit <- svm(Revenue ~ ., data = online_shopper_training, kernel = "radial",gamma=0.03125,cost=2)
    
    end_time <- Sys.time()
    total_time <- end_time - start_time
    print(total_time)
    
    ygrid <- predict(svmfit, online_shopper_testing)
    test_predict <- table(ygrid, online_shopper_testing[, "Revenue"])
    
    #it gives names to table
    #rownames(test_predict) <- paste("Actual", rownames(test_predict), sep = ":")
    #colnames(test_predict) <- paste("Predicted", colnames(test_predict), sep = ":")
    #print(test_predict)
    
    #It is stored to values of all metrics in the matrix
    matrixAcc[5,fold] <-  sum(diag(test_predict)) / sum(test_predict)
    matrixPre[5,fold] <-  test_predict[1,1]/sum(test_predict[,1])
    matrixRecall[5,fold] <- test_predict[1,1]/sum(test_predict[1,])
    matrixF1score[5,fold] <- 2 * (matrixPre[5,fold] * matrixRecall[5,fold]) / (matrixPre[5,fold]+matrixRecall[5,fold])
    matrixAUC[5,fold] <- fun.auc(ifelse(ygrid=="TRUE",1,0),as.factor(as.logical(online_shopper_testing$Revenue)))
    
    
    svmRadial_metricsVal<- c( matrixAcc[5,fold],matrixPre[5,fold],matrixRecall[5,fold],matrixF1score[5,fold],matrixAUC[5,fold])
    barplot(svmRadial_metricsVal,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Accuracy","Precision","Recall","f1_score","AUC"),col=blues9)
    
    
  
  # ##Desicion Tree(ID3)------------------
   
    set.seed(1815850)
      start_time <- Sys.time()
      
      fit2 <- tree(Revenue ~ ., data = online_shopper_training)
      
      end_time <- Sys.time()
      total_time <- end_time - start_time
      print(total_time)
      yhat.tree<-predict(fit2, online_shopper_testing,type = "class")
      test_predict <- table(yhat.tree, online_shopper_testing[, "Revenue"])
      
      #it gives names to table
      #rownames(test_predict) <- paste("Actual", rownames(test_predict), sep = ":")
      #colnames(test_predict) <- paste("Predicted", colnames(test_predict), sep = ":")
      
      #It is stored to values of all metrics in the matrix
      matrixAcc[6,fold] <-  sum(diag(test_predict)) / sum(test_predict)
      matrixPre[6,fold] <-  test_predict[1,1]/sum(test_predict[,1])
      matrixRecall[6,fold] <- test_predict[1,1]/sum(test_predict[1,])
      matrixF1score[6,fold] <- 2 * (matrixPre[6,fold] * matrixRecall[6,fold]) / (matrixRecall[6,fold])
      matrixAUC[6,fold] <- fun.auc(ifelse(yhat.tree=="TRUE",1,0),as.factor(as.logical(online_shopper_testing$Revenue)))
      
      ID3_metricsVal<- c( matrixAcc[6,fold],matrixPre[6,fold],matrixRecall[6,fold],matrixF1score[6,fold],matrixAUC[6,fold])
      barplot(ID3_metricsVal,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Accuracy","Precision","Recall","f1_score","AUC"),col=blues9)
      
      
  
 
  ##Random Forest------
      
      
        set.seed(50600)
        start_time <- Sys.time()
        
        # Create the forest.
        output.forest <- randomForest(Revenue ~ ., 
                                      data = online_shopper_training,mtry=5,ntrees=500)
        end_time <- Sys.time()
        total_time <- end_time - start_time
        print(total_time)
        yhat.bag=predict(output.forest,newdata=online_shopper_testing,type="class")
        test_predict <- table(yhat.bag, online_shopper_testing[, "Revenue"])
        
        #it gives names to table
        #rownames(test_predict) <- paste("Actual", rownames(test_predict), sep = ":")
        #colnames(test_predict) <- paste("Predicted", colnames(test_predict), sep = ":")
        #print(test_predict)
      
        #It is stored to values of all metrics in the matrix
        matrixAcc[7,fold] <-  sum(diag(test_predict)) / sum(test_predict)
        matrixPre[7,fold] <-  test_predict[1,1]/sum(test_predict[,1])
        matrixRecall[7,fold] <- test_predict[1,1]/sum(test_predict[1,])
        matrixF1score[7,fold] <- 2 * (matrixPre[7,fold] * matrixRecall[7,fold]) / (matrixRecall[7,fold])
        matrixAUC[7,fold] <- fun.auc(ifelse(ygrid=="TRUE",1,0),as.factor(as.logical(online_shopper_testing$Revenue)))
        
        randomForest_metricsVal<- c( matrixAcc[7,fold],matrixPre[7,fold],matrixRecall[7,fold],matrixF1score[7,fold],matrixAUC[7,fold])
        barplot(randomForest_metricsVal,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Acc","Pre","Recall","fScore","AUC"),col=blues9)
        
      
      #Plot the error rates or MSE of a randomForest object
      plot(output.forest)
      
      # Importance of each predictor.
      print(importance(output.forest,type = 2)) 
      # graphic of importance of predictors.
      varImpPlot(output.forest)
      

}#CROSS VALÝDATÝON
#When cross validation finished , starts prints
  for(k in 1:7){
    nameModel(k)
    printMetricResult(k)
  }
}#undersampling
#You must run ANN algortihm seperately. Because ýt has different number of neuron and hidden layer combinations.
##Artifical Neural Network--------------------------------------------------------------------------------------

prepared_onlineShopper$Revenue <- as.integer(as.logical(prepared_onlineShopper$Revenue))
online_shopper_training$Revenue <- as.integer(as.logical(online_shopper_training$Revenue))
online_shopper_testing$Revenue <- as.integer(as.logical(online_shopper_testing$Revenue))
#nn <- neuralnet(shopper_df_train$Revenue~., data=shopper_df_train, hidden=c(5,2), stepmax=1e6, linear.output=FALSE, threshold=0.01)
start_time <- Sys.time()
nn <- neuralnet(online_shopper_training$Revenue~., data=online_shopper_training, hidden=1,
                stepmax = 1e6,
                learningrate = 0.1,linear.output=FALSE, threshold=0.01)
end_time <- Sys.time()
total_time <- end_time - start_time
print(total_time)
#nn$result.matrix
#plot(nn)

#Test the resulting output
nn.results <- compute(nn, online_shopper_testing)
results <- data.frame(actual = online_shopper_testing$Revenue, prediction = nn.results$net.result)

roundedresults<-sapply(results,round,digits=0)
rm(roundedresultsdf)
roundedresultsdf <- data.frame(roundedresults)

attach(roundedresultsdf)

conf_matrix <- table(online_shopper_testing$Revenue,prediction)
rownames(conf_matrix) <- c("Actual: False","Actual: True")
colnames(conf_matrix) <- c("Prediction: False","Prediction: True")
print(conf_matrix)

# Number of correct predictions
numCorrect <- length(which(online_shopper_testing$Revenue==prediction))
# Number of misclassifications
numMiss <- length(which(online_shopper_testing$Revenue!=prediction))
# Rate of correct classifications
rate <- numCorrect/length(online_shopper_testing$Revenue) * 100;
cat("Accuracy: ",rate)
