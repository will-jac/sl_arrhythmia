library(dplyr)
library(data.tree)

###################################################################################################
# ID3 Function
###################################################################################################
ID3=function(examples, attributes, types, count=1, target="Arrhythmia")
{
  print(count) ############## debugging 
  
  root <- Node$new()
  root$name = "new"
  root$threshold = -1000
  root$type = "root"
  root$label <- as.integer(names(sort(table(examples[,target]), decreasing = TRUE))[1]) #label root with most common label in target
  root$reason <- "nothing"  ####################### debugging
  
  if (length(unique(examples[,target]))==1){
    #root$name <- as.character(examples[1,target])
    #root$label <- examples[1,target]
    root$name <- as.character(root$label)
    root$reason <- "All examples had the same label"  ####################### debugging
    #print("All examples had the same label") #################### debugging
    return(root)
  }
  
  if (length(attributes)==0)
  {
    #lab = names(sort(table(examples[,target]), decreasing = TRUE))[1]
    #root$name <- as.character(lab)
    #root$label <- as.integer(lab)
    root$name <- as.character(root$label)
    root$reason <- "There were no attributes left"  ####################### debugging
    #print("There were no attributes left") ####################### debugging
    return(root)
  }
  
  #Figure out which attribute is best
  result = max_info_gain(examples=examples, attributes=attributes, types=types, target=target)
  A = result$attribute
  threshold = result$threshold
  
  root$name <- A
  root$threshold = threshold
  
  print(A) ############################################# debugging
  
  #remove A from attributes
  sub_attributes = attributes[attributes %in% A == FALSE]
  
  #Find subset of the examples according to A
  less_than_examples = examples %>% filter(get(A) <= threshold)
  greater_than_examples = examples %>% filter(get(A) > threshold)

  if(nrow(less_than_examples) == 0 | nrow(greater_than_examples) == 0)
  {
    #lab = names(sort(table(examples[,target]), decreasing = TRUE))[1]
    #root$name <- as.character(lab)
    #root$label <- as.integer(lab)
    root$name <- as.character(root$label)
    root$reason <- "threshold was at the edge of the possible values (i.e. no decision, just one set of values)" ####################### debugging
    ####################################################################################### debugging
    #print("threshold was at the edge of the possible values (i.e. no decision, just one set of values)")
    return(root)
  }
  
  #for 0 (or less than threshold)
  child = ID3(less_than_examples, attributes=sub_attributes, types=types, count=count+1)
  child$type = "<="
  root$AddChildNode(child)
  
  #for 1 (or greater than threshold)
  child = ID3(greater_than_examples, sub_attributes, types=types, count=count+1)
  child$type = ">"
  root$AddChildNode(child)
  
  #print(root, "threshold", "type", "label", "reason") ##################################################debugging
  
  if(count == 1)
  {
    root$Set(id = 1:root$totalCount, traversal = "post-order")
  }
  
  return(root)
}






###################################################################################################
#   max info gain 
###################################################################################################
max_info_gain=function(examples, attributes, types, target = "Arrhythmia"){
  entropy_tot = entropy(examples[,target])
  n_exs = nrow(examples)
  
  best_A = "NA"
  max_gain = -1
  threshold = 0.5
  
  for(A in attributes)
  {
    #Find info gain
    if(types[A]=="binary")
    {
      gain = info_gain(examples=examples, A=A, threshold=0.5, n_exs=n_exs, entropy_tot=entropy_tot, target=target)
      t = 0.5
    }
    else
    {
      result = info_gain_threshold(examples=examples, A=A, n_exs=n_exs, entropy_tot=entropy_tot, target=target)
      gain = result$info_gain
      t = result$threshold
    }
    
    #print(paste(A, ": ", gain, sep=""))  ###############################################################debugging
    
    
    #compare to current maximum
    if(gain > max_gain)
    {
      max_gain = gain
      best_A = A
      threshold = t
    }
  }
  
  
  return(list(attribute = best_A, threshold=threshold, info_gain = max_gain))
}





###################################################################################################
#  info gain
###################################################################################################
info_gain=function(examples, A, threshold, n_exs, entropy_tot, target="Arrhythmia"){
  
  sub_examples = dplyr::filter(examples, get(A)<=threshold)$Arrhythmia #############################FIX IF REUSE CODE
  sum = ((length(sub_examples)/n_exs)*entropy(sub_examples))
  
  sub_examples = dplyr::filter(examples, get(A)>threshold)$Arrhythmia #############################FIX IF REUSE CODE
  sum = sum+ ((length(sub_examples)/n_exs)*entropy(sub_examples))
  
  gain = entropy_tot-sum
  return(gain)
}




###################################################################################################
#entropy
###################################################################################################
entropy=function(target_vals){
  prop_labels = (table(target_vals)/length(target_vals))
  return(-sum(prop_labels*log2(prop_labels)))
}





###################################################################################################
#   info gain (real valued attributes)
###################################################################################################
info_gain_threshold=function(examples, A, n_exs, entropy_tot, target="Arrhythmia"){
  sorted = dplyr::arrange(examples, A)

  label = sorted[1,target]
  switches = c()
  s = 1

  for(i in 1:nrow(sorted))
  {
    if(label != sorted[i,target])
    {
      switches[s]=i
      s=s+1
      label = sorted[i,target]
    }
  }
  
  #print(A) #################################################################debugging
  #print("switches:") #################################################################debugging
  #print(switches) #################################################################debugging

  threshold = (sorted[switches[1],A]+sorted[(switches[1]-1),A])/2
  max_gain = -1

  for(s in switches)
  {
    t = (sorted[s,A]+sorted[(s-1),A])/2
    gain = info_gain(examples=sorted, A, threshold=t, n_exs, entropy_tot, target=target)

    if(gain > max_gain)
    {
      max_gain = gain
      threshold = t
    }
  }
  
  #print("values:") #################################################################debugging
  #print(examples[,A]) #################################################################debugging
  
  #print("threshold:") #################################################################debugging
  #print(threshold) #################################################################debugging
  
  return(list(info_gain = max_gain, threshold = threshold))
}






###################################################################################################
# make predictions with the tree
###################################################################################################
mypredict=function(examples, root, nodeID=-1, target="Arrhythmia"){
  predictions = c()
  
  #make a prediction for each row
  for(i in 1:nrow(examples))
  {
    node = root #start at the root
    
    #until leaf node (or node whose children are being pruned) is reached 
    while((!isLeaf(node)) & (node$id != nodeID))
    {
      #if node has only one child, move to that child
      if(length(node$children)==1)
        node = node$children[[1]]
      else{
        if(examples[i,node$name]<=node$threshold){
          #go to "less than or equals" child node
          node = node$children[[1]]
        }
        else{
          #go to "greater than" child node
          node = node$children[[2]]
        }
      }
    }
    predictions[i] = node$label
  }
  
  num_right = 0
  for(i in 1:length(predictions))
  {
    if(predictions[i] == examples[i,target])
      num_right = num_right+1
  }
  
  return(list(predictions = predictions, correct_vals = examples[,target], accuracy = (num_right/length(predictions))))
}








###################################################################################################
# prune the trees
###################################################################################################
pruning=function(root, validation_data, target="Arrhythmia"){
  max_acc = mypredict(validation_data, root)$accuracy
  
  n = root$totalCount
  continue = TRUE
  
  while(continue)
  {
    #Find what the accuracy would be if any given node's children were removed
    pruning_accuracies = rep(0, n)
    
    IDs = root$Get('id', filterFun = function(x) !x$isLeaf)
    
    for(id in IDs)
    {
      pruning_accuracies[id] = mypredict(examples=validation_data, root=root, nodeID=id, target=target)$accuracy
    }
    
    #print(pruning_accuracies) ######################################################debugging
    
    #If accuracy can be improved by pruning a node's kids, prune 
    if(max(pruning_accuracies)>=max_acc)
    {
      max_acc = max(pruning_accuracies)
      
      ID = which(pruning_accuracies == max_acc)[[1]]
      
      #change the node's name to its label
      #lab = root$Get('label', filterFun = function(x) x$id=ID)
      #root$Set(name=as.character(lab), filterFun = function(x) x$id=ID)
      
      childids = root$children[[1]]$Get('id', filterFun = function(x) x$parent$id == ID)
      if(is.null(childids))
        childids = root$children[[2]]$Get('id', filterFun = function(x) x$parent$id == ID)
      
      #print(childids) ####################################################################debugging
      
      for(i in childids){
        Prune(root, pruneFun = function(x) x$id != i)
      }
    }
    else
    {
      continue=FALSE #if accuracy cannot be improved by pruning, stop pruning
    }
  }
  
  #return the new accuracy of the tree
  return(max_acc)
}





###################################################################################################
#    "main" method
###################################################################################################
setwd("~/Dropbox/Classwork Fall 2020/Machine Learning/SLProject/sl-arrhythmias")
data <- read.csv("arrhythmia.csv")

#data <- filter(data, HeartRate!='?')
#data <- mutate(data, HeartRate = as.integer(HeartRate))

#### remove the columns with missing values
drop <- c("T","P","QRST","J", "HeartRate")
data <- select(data, -which(names(data) %in% drop))
rm(drop)


#remove target variable from the attributes
feats <- names(data)[names(data) %in% "Arrhythmia" == FALSE]

#get types for each attribute
types = c()
for(i in 1:length(feats))
{
  poss_vals = unique(data[,feats[i]])
  #if binary
  if(length(poss_vals)<=2 &(poss_vals[1]==0 | poss_vals[1]==1))
  {
    types[i] = "binary"
  }
  else
  {
    types[i] = "real"
  }
}
rm(i)
rm(poss_vals)

names(types)=feats

#find just the binary attributes
# binary_feats = c()
# i=1
# for(str in feats)
# {
#   if(types[str]=="binary")
#   {
#     binary_feats[i] = str
#     i = i+1
#   }
# }
# rm(str)
# rm(i)


#### split data into testing, training, and validation
num_test = floor(nrow(data)*.2)

#random index order
set.seed(25)
ind = sample(nrow(data), nrow(data), replace = FALSE)

test <- data[ind[1:num_test], ]
valid <- data[ind[(num_test+1):(2*num_test)],]
train <- data[ind[((2*num_test)+1):nrow(data)],]

rm(ind)
rm(num_test)
#rm(test)

#################
#Train the model
#################s
root <- ID3(train, feats, types)

#print(root, "id", "threshold", "type", "label", "reason")
print("Tree with usual data:")
mypredict(train, root)
mypredict(valid, root)
mypredict(test, root)

#save and clone the tree before pruning!!!!!!!!!
saveRDS(root, file="fullrealtree_withHR")

pruning(root, valid)

saveRDS(root, file="fullrealtree_withHR_pruned")

print("Tree with usual data after pruning:")
mypredict(train, root)
mypredict(valid, root)
mypredict(test, root)

#################
#Train the model on all training data I guess 
#################s
train = nrow(rbind(train, valid))

root <- ID3(train, feats, types)

#print(root, "id", "threshold", "type", "label", "reason")
print("Tree with all training data:")
mypredict(train, root)
mypredict(test, root)

#save and clone the tree before pruning!!!!!!!!!
saveRDS(root, file="fullrealtree_withHR_2")

pruning(root, test)

saveRDS(root, file="fullrealtree_withHR_2_pruned")

print("Tree with usual data after pruning:")
mypredict(train, root)
mypredict(test, root)

######
#cross validation
######

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
  
  #nvalid = floor(nrow(train)*0.25)
  #set.seed(i)
  #ind = sample(nrow(train), nrow(train), replace = FALSE)
  #valid = train[ind[1:nvalid],]
  #train = train[ind[nvalid+1:nrow(train)],]
  
  root <- ID3(train, feats, types)
  saveRDS(root, file=paste("Xvalidationtree", i, sep="_"))
  print(root, "id", "threshold", "type", "label", "reason")
  
  print(paste("Tree ", i, ":", sep=""))
  mypredict(train, root)
  #mypredict(valid, root)
  mypredict(test, root)
  
  pruning(root, test)
  saveRDS(root, file=paste("Xvalidationtree", i, "pruned", sep="_"))
  print(root, "id", "threshold", "type", "label", "reason")
  
  print(paste("Tree ", i, " after pruning:", sep=""))
  mypredict(train, root)
  #mypredict(valid, root)
  mypredict(test, root)
}
  