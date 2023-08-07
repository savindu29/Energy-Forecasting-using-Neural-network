# Libraries
library(neuralnet)
library(readxl)
library(dplyr)  # to use lag function
library(Metrics)  # to use RMSE function
library(stats)


# Reading data set
uow_consumption <- read_excel("uow_consumption.xlsx")
colnames(uow_consumption)<-c("Date","18:00","19:00","20:00")  # Rename column names
uow_consumption <- uow_consumption[2:4]



# Creating a data frames including "20:00" column
data <- as.data.frame(uow_consumption$"20:00")


#Checking missing values
sum(is.na(data))


# Calculating lagged values
t_1 <- lag(data,1)
t_2 <- lag(data,2)
t_3 <- lag(data,3)
t_4 <- lag(data,4)
t_7 <- lag(data,7)



# Creating I/O Matrixes
M1 <- cbind(t_1,data)
M2 <- cbind(t_1,t_2,data)
M3 <- cbind(t_1,t_2,t_3,data)
M4 <- cbind(t_1,t_2,t_3,t_4,data)
M5 <- cbind(t_1,t_2,t_3,t_4,t_7,data)



# Rename column names in I/O Matrixes
colnames(M1)<-c("t-1","output")
colnames(M2)<-c("t-1","t-2","output")
colnames(M3)<-c("t-1","t-2","t-3","output")
colnames(M4)<-c("t-1","t-2","t-3","t-4","output")
colnames(M5)<-c("t-1","t-2","t-3","t-4","t-7","output")



# Remove missing values
M1 <- M1[complete.cases(M1),]
M2 <- M2[complete.cases(M2),]
M3 <- M3[complete.cases(M3),]
M4 <- M4[complete.cases(M4),]
M5 <- M5[complete.cases(M5),]



# Checking outliers
boxplot(M1)
boxplot(M2)
boxplot(M3)
boxplot(M4)
boxplot(M5)




# min max normalization 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


# De-Normalization
unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}




# Apply normalization
M1_norm <- as.data.frame(lapply(M1, normalize))
M2_norm <- as.data.frame(lapply(M2, normalize))
M3_norm <- as.data.frame(lapply(M3, normalize))
M4_norm <- as.data.frame(lapply(M4, normalize))
M5_norm <- as.data.frame(lapply(M5, normalize))




#Creating training and testing data
M1_train_norm <- M1_norm[1:380, ]
M2_train_norm <- M2_norm[1:380, ]
M3_train_norm <- M3_norm[1:380, ]
M4_train_norm <- M4_norm[1:380, ]
M5_train_norm <- M5_norm[1:380, ]

M1_test_norm <- M1_norm[381:469, ]
M2_test_norm <- M2_norm[381:468, ]
M3_test_norm <- M3_norm[381:467, ]
M4_test_norm <- M4_norm[381:466, ]
M5_test_norm <- M5_norm[381:463, ]





# Evaluation function
evaluate <- function(actual,prediction){
  rmse <- rmse(actual = actual,predicted = prediction)
  mae <- mae(actual = actual, predicted = prediction)
  mape <- mape(actual = actual, predicted = prediction)
  smape <- smape(actual = actual, predicted = prediction)
  
  df <- data.frame(stat.indices = c("RMSE","MAE","MAPE","sMAPE"),values = c(rmse,mae,mape,smape))
  return(df)
  
}



set.seed(20)

# M1
original_train_outputs_M1 <- M1[1:380,"output"]
original_test_outputs_M1 <- M1[381:469,"output"]

min_output_M1 <- min(original_train_outputs_M1)
max_output_M1 <- max(original_train_outputs_M1)

# neural network code/training
model1 <- neuralnet(output ~ t.1, data = M1_train_norm, hidden = 6, act.fct = 'logistic', linear.output = T)
plot(model1)
model1_results <- neuralnet::compute(model1,M1_test_norm[1])
predicted_output_model1 <- unnormalize(model1_results$net.result,min_output_M1,max_output_M1) #obtain predicted output
evaluate(original_test_outputs_M1,predicted_output_model1)  # evaluation

# neural network code/training
model2 <- neuralnet(output ~ t.1, data = M1_train_norm, hidden = c(7,4), act.fct = 'logistic', linear.output = T)
plot(model2)
model2_results <- neuralnet::compute(model2,M1_test_norm[1])
predicted_output_model2 <- unnormalize(model2_results$net.result,min_output_M1,max_output_M1) #obtain predicted output
evaluate(original_test_outputs_M1,predicted_output_model2)  # evaluation

# neural network code/training
model3 <- neuralnet(output ~ t.1, data = M1_train_norm, hidden = c(9,4), act.fct = 'logistic', linear.output = F)
plot(model3)
model3_results <- neuralnet::compute(model3,M1_test_norm[1])
predicted_output_model3 <- unnormalize(model3_results$net.result,min_output_M1,max_output_M1) #obtain predicted output
evaluate(original_test_outputs_M1,predicted_output_model3)  # evaluation



#M2
original_train_outputs_M2 <- M2[1:380,"output"]
original_test_outputs_M2 <- M2[381:468,"output"]

min_output_M2 <- min(original_train_outputs_M2)
max_output_M2 <- max(original_train_outputs_M2)

# neural network code/training
model4 <- neuralnet(output ~ t.1 + t.2, data = M2_train_norm, hidden = 10, act.fct = 'logistic', linear.output = T)
plot(model4)
model4_results <- neuralnet::compute(model4,M2_test_norm[1:2])
predicted_output_model4 <- unnormalize(model4_results$net.result,min_output_M2,max_output_M2) #obtain predicted output
evaluate(original_test_outputs_M2,predicted_output_model4)  # evaluation

# neural network code/training
model5 <- neuralnet(output ~ t.1 + t.2, data = M2_train_norm, hidden = c(8,6), act.fct = 'logistic', linear.output = T)
plot(model5)
model5_results <- neuralnet::compute(model5,M2_test_norm[1:2])
predicted_output_model5 <- unnormalize(model5_results$net.result,min_output_M2,max_output_M2) #obtain predicted output
evaluate(original_test_outputs_M2,predicted_output_model5)  # evaluation


# neural network code/training
model6 <- neuralnet(output ~ t.1 + t.2, data = M2_train_norm, hidden = c(8,6), act.fct = 'tanh', linear.output = T)
plot(model6)
model6_results <- neuralnet::compute(model6,M2_test_norm[1:2])
predicted_output_model6 <- unnormalize(model6_results$net.result,min_output_M2,max_output_M2) #obtain predicted output
evaluate(original_test_outputs_M2,predicted_output_model6)  # evaluation



#M3
original_train_outputs_M3 <- M3[1:380,"output"]
original_test_outputs_M3 <- M3[381:467,"output"]

min_output_M3 <- min(original_train_outputs_M3)
max_output_M3 <- max(original_train_outputs_M3)

# neural network code/training
model7 <- neuralnet(output ~ t.1 + t.2 + t.3, data = M3_train_norm, hidden = 10, act.fct = 'logistic', linear.output = T)
plot(model7)
model7_results <- neuralnet::compute(model7,M3_test_norm[1:3])
predicted_output_model7 <- unnormalize(model7_results$net.result,min_output_M3,max_output_M3) #obtain predicted output
evaluate(original_test_outputs_M3,predicted_output_model7)  # evaluation

# neural network code/training
model8 <- neuralnet(output ~ t.1 + t.2 + t.3, data = M3_train_norm, hidden = 10, act.fct = 'logistic', linear.output = F)
plot(model8)
model8_results <- neuralnet::compute(model8,M3_test_norm[1:3])
predicted_output_model8 <- unnormalize(model8_results$net.result,min_output_M3,max_output_M3) #obtain predicted output
evaluate(original_test_outputs_M3,predicted_output_model8)  # evaluation

# neural network code/training
model9 <- neuralnet(output ~ t.1 + t.2 + t.3, data = M3_train_norm, hidden = c(10,6), act.fct = 'logistic', linear.output = T)
plot(model9)
model9_results <- neuralnet::compute(model9,M3_test_norm[1:3])
predicted_output_model9 <- unnormalize(model9_results$net.result,min_output_M3,max_output_M3) #obtain predicted output
evaluate(original_test_outputs_M3,predicted_output_model9)  # evaluation




#M4
original_train_outputs_M4 <- M4[1:380,"output"]
original_test_outputs_M4 <- M4[381:466,"output"]

min_output_M4 <- min(original_train_outputs_M4)
max_output_M4 <- max(original_train_outputs_M4)

# neural network code/training
model10 <- neuralnet(output ~ t.1 + t.2 + t.3 + t.4, data = M4_train_norm, hidden = 10, act.fct = 'logistic', linear.output = T)
plot(model10)
model10_results <- neuralnet::compute(model10,M4_test_norm[1:4])
predicted_output_model10 <- unnormalize(model10_results$net.result,min_output_M4,max_output_M4) #obtain predicted output
evaluate(original_test_outputs_M4,predicted_output_model10)  # evaluation

# neural network code/training
model11 <- neuralnet(output ~ t.1 + t.2 + t.3 + t.4, data = M4_train_norm, hidden = c(8,4), act.fct = 'logistic', linear.output = T)
plot(model11)
model11_results <- neuralnet::compute(model11,M4_test_norm[1:4])
predicted_output_model11 <- unnormalize(model11_results$net.result,min_output_M4,max_output_M4) #obtain predicted output
evaluate(original_test_outputs_M4,predicted_output_model11)  # evaluation

# neural network code/training
model12 <- neuralnet(output ~ t.1 + t.2 + t.3 + t.4, data = M4_train_norm, hidden = c(8,4), act.fct = 'tanh', linear.output = T)
plot(model12)
model12_results <- neuralnet::compute(model12,M4_test_norm[1:4])
predicted_output_model12 <- unnormalize(model12_results$net.result,min_output_M4,max_output_M4) #obtain predicted output
evaluate(original_test_outputs_M4,predicted_output_model12)  # evaluation



#M5
original_train_outputs_M5 <- M5[1:380,"output"]
original_test_outputs_M5 <- M5[381:463,"output"]

min_output_M5 <- min(original_train_outputs_M5)
max_output_M5 <- max(original_train_outputs_M5)

# neural network code/training
model13 <- neuralnet(output ~ t.1 + t.2 + t.3 + t.4 + t.7, data = M5_train_norm, hidden = c(8,4), act.fct = 'logistic', linear.output = T)
plot(model13)
model13_results <- neuralnet::compute(model13,M5_test_norm[1:5])
predicted_output_model13 <- unnormalize(model13_results$net.result,min_output_M5,max_output_M5) #obtain predicted output
evaluate(original_test_outputs_M5,predicted_output_model13)  # evaluation

# neural network code/training
model14 <- neuralnet(output ~ t.1 + t.2 + t.3 + t.4 + t.7, data = M5_train_norm, hidden = c(10,6), act.fct = 'logistic', linear.output = T)
plot(model14)
model14_results <- neuralnet::compute(model14,M5_test_norm[1:5])
predicted_output_model14 <- unnormalize(model14_results$net.result,min_output_M5,max_output_M5) #obtain predicted output
evaluate(original_test_outputs_M5,predicted_output_model14)  # evaluation

# neural network code/training
model15 <- neuralnet(output ~ t.1 + t.2 + t.3 + t.4 + t.7, data = M5_train_norm, hidden = c(10,6), act.fct = 'logistic', linear.output = F)
plot(model15)
model15_results <- neuralnet::compute(model15,M5_test_norm[1:5])
predicted_output_model15 <- unnormalize(model15_results$net.result,min_output_M5,max_output_M5) #obtain predicted output
evaluate(original_test_outputs_M5,predicted_output_model15)  # evaluation






# model 15 - graphical representation 
par(mfrow=c(1,1))
plot(original_test_outputs_M5, predicted_output_model15 ,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(a=0, b=1, h=90, v=90)
