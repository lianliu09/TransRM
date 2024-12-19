library(randomForest)
library("caret")
library(pROC)
#k: k fold
#data: the data used for cross validataion
#dir: the directory where the results are saved
#filename: the result files name
cross_validation_rf <- function(k=5,data){
  
  #train
  n=length(names(data))
  folds<-createFolds(y=data$Target_label,k=k) #根据training的laber-Species把数据集切分成k等份
  TPR=c(length=k)
  FPR=c(length=k)
  MCC=c(length=k)
  F1=c(length=k)
  ACC=c(length=k)
  precision=c(length=k)
  recall=c(length=k)
  ROCArea=c(length=k)
  AUPR=c(length=k)
  #PRCArea=c(length=k)
  prob=c(length=312)
  for(i in 1:k){
    print(paste(i,"-fold"))
    index=1:nrow(data)
    index_train=sample(index[-folds[[i]]])
    train<-data[index_train,]
    index_test=sample(index[folds[[i]]])
    test<-data[index_test,]
    #    y_test<-y_label[index_test]
    rf <- randomForest(Target_label ~ ., data=train)
    classification=predict(rf,newdata=test)
    pro=predict(rf,newdata=test,type="prob")
    prob[index_test]=predict(rf,newdata=test,type="prob")
    TP <- as.numeric(sum(classification=="positive" & test$Target_label=="positive"))
    FP <- as.numeric(sum(classification=="positive" & test$Target_label=="negative"))
    TN <- as.numeric(sum(classification=="negative" & test$Target_label=="negative"))
    FN <- as.numeric(sum(classification=="negative" & test$Target_label=="positive"))
    TPR[i]=TP/(TP+FN)
    FPR[i]=TN/(TN+FP)
    MCC[i]=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    F1[i]=(2*TP)/(2*TP+FP+FN)
    ACC[i]=(TP+TN)/(TP+TN+FP+FN)
    precision[i]=TP/(TP+FP)
    recall[i]=TP/(TP+FN)
    ROCArea[i]=auc(roc(test$Target_label,pro[,2]))
    pred <-prediction(pro[,2],test$Target_label)
    perf <- performance(pred,"prec","rec")
    plot(perf, col='blue',lty=2)
    aucpr <- performance(pred,'aucpr')
    AUPR[i] = unlist(slot(aucpr,"y.values"))
    #    AUPR[i]=AUC(obs=y_test,pred=pro[,2],curve = "PR", simplif=TRUE, main = "PR curve")
  }
  result=data.frame(mean(TPR),mean(FPR),mean(ACC),mean(MCC),mean(F1),mean(precision),mean(recall),mean(ROCArea),mean(AUPR))
  colnames(result)<-c("Sn","Sp","ACC","MCC","F1","Pre","Recall","AUC","AUPR")
  # file1=paste(dir,paste(filename,".result.csv"),sep = '/')
  # write.csv(result,paste(dir,paste(filename,k,".result.rf.csv"),sep = '/'),row.names = F)
  #  write.csv(prob,"E:/m6A-circRNA/data/test_data/data_m6a/prob_rf.csv",row.names = F)
  
  return(result)
}
library(randomForest)
library("caret")
library(pROC)
library(ROCR)

path = paste("E:/m6A-circRNA/paper_data/result/bagfeature_ndata",".csv",sep="")
data = read.csv(path,header=T)
y=c(rep("positive",156),rep("negative",156))
data=cbind(data,y)

colnames(data)=c(1:(ncol(data)-1),"Target_label")
write.csv(data,"E:/m6A-circRNA/paper_data/result/features_all.csv",row.names = F)
data = read.csv("E:/m6A-circRNA/paper_data/result/features_all.csv",header=T)
#data=read.csv("E:/m6A-circRNA/data/test_data/data_m6a/features_mid.csv",header = T)
# result_R=as.data.frame(array(,dim=c(10,9)))
# result_S=as.data.frame(array(,dim=c(10,9)))
# result_K=as.data.frame(array(,dim=c(10,9)))
# result_L=as.data.frame(array(,dim=c(10,9)))
# result_X=as.data.frame(array(,dim=c(10,9)))
result_R=cross_validation_rf(k=5,data=data)