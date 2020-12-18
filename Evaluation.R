#### 10 fold cross validation results ####

predictions = matrix(rep(0, nrow(data)*10), nrow = 10, dimnames = 
                       list(c("tree1", "tree2","tree3","tree4","tree5","tree6","tree7","tree8","tree9","tree10")))


for(i in 1:10)
{
  root <- readRDS(paste("Xvalidationtree_", i, "_pruned", sep=""))
  predictions[i,] = mypredict(data, root)$predictions
}

pred_total = c()

for(i in 1:nrow(data))
{
  pred_total[i] = as.integer(names(sort(table(predictions[,i]),decreasing=TRUE))[1])
}



stats = matrix(rep(0, 6*10), nrow = 10, dimnames = 
                       list(c("tree1", "tree2","tree3","tree4","tree5","tree6","tree7","tree8","tree9","tree10"),
                            c("Training Accuracy Before Prune","Testing Accuracy Before Prune",
                              "Training Accuracy After Prune", "Testing Accuracy After Prune",
                              "Sensitivity After Prune", "Specificity After Prune")))

num = 45
lwr = c()
upr = c()
for(i in 1:10)
{
  lwr[i] = num*(i-1)+1
  upr[i] = num*(i-1)+num
}
upr[10] = upr[10]+1



for(i in 1:10)
{
  test = data[lwr[i]:upr[i],]
  if(i!=1 & i!=10)
    train = data[c(1:(lwr[i]-1),(upr[i]+1):451),]
  else
  {
    if(i==1)
      train = data[(upr[1]+1):451,]
    else
      train = data[1:(lwr[10]-1),]
  }
  
  root <- readRDS(paste("Xvalidationtree_", i, sep=""))
  stats[i,1] = mypredict(train, root)$accuracy
  stats[i,2] = mypredict(test, root)$accuracy
  root <- readRDS(paste("Xvalidationtree_", i, "_pruned", sep=""))
  stats[i,3] = mypredict(train, root)$accuracy
  stats[i,4] = mypredict(test, root)$accuracy
  
  #stats[i,5] = cmat1[1,1]/(cmat1[1,1]+cmat1[1,2])
  #stats[i,6] = cmat1[2,2]/(cmat1[2,2]+cmat1[2,1])
}








df = test

pred = mypredict(df, root)$predictions
real = df[,"Arrhythmia"]

###### make a confusion matrix for just presence/absence of any kind of arrhythmia

cmat1 = matrix(c(0,0,0,0), nrow=2, dimnames = list(c("Truly Pos.", "Truly Neg."),c("Pred. Pos.", "Pred. Neg.")))

for(i in 1:length(pred))
{
  if(pred[i]==1)
  {
    if(real[i]==1)
      cmat1[2,2] = cmat1[2,2]+1
    else
      cmat1[1,2] = cmat1[1,2]+1
  }
  else
    if(real[i]==1)
      cmat1[2,1] = cmat1[2,1]+1
    else
      cmat1[1,1] = cmat1[1,1]+1
}

cmat1

###### make a giant confusion matrix 
vals = c(1:16)
m = length(vals)
cmat2 = matrix(rep(0, m*m), nrow=m, dimnames = list(paste("Truly", vals, sep=" "),
                                                   paste("Pred", vals, sep=" ")))

for(i in 1:length(pred))
{
  cmat2[real[i],pred[i]] = cmat2[real[i],pred[i]]+1
}

cmat2

accuracy = mypredict(df, root)$accuracy
accuracy

risk = 1-accuracy
risk

#overall true positive rate
sensitivity = cmat1[1,1]/(cmat1[1,1]+cmat1[1,2])
sensitivity

#overall proportion of positives that were true
precision = cmat1[1,1]/(cmat1[1,1]+cmat1[2,1])
precision

#overall true negative rate
specificity = cmat1[2,2]/(cmat1[2,2]+cmat1[2,1])
specificity

