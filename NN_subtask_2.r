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


#Checking missing values
sum(is.na(uow_consumption))




# Calculating lagged values for 20th hour
t20_1 <- lag(uow_consumption[3],1)
t20_2 <- lag(uow_consumption[3],2)
t20_3 <- lag(uow_consumption[3],3)
t20_4 <- lag(uow_consumption[3],4)
t20_7 <- lag(uow_consumption[3],7)

# Calculating lagged values for 19th hour
t19_1 <- lag(uow_consumption[2],1)
t19_2 <- lag(uow_consumption[2],2)
t19_3 <- lag(uow_consumption[2],3)
t19_4 <- lag(uow_consumption[2],4)
t19_7 <- lag(uow_consumption[2],7)

# Calculating lagged values for 18th hour
t18_1 <- lag(uow_consumption[1],1)
t18_2 <- lag(uow_consumption[1],2)
t18_3 <- lag(uow_consumption[1],3)
t18_4 <- lag(uow_consumption[1],4)
t18_7 <- lag(uow_consumption[1],7)





# Creating I/O Matrixes
M1 <- cbind(t18_1,t19_1,t20_1,uow_consumption[3])
M2 <- cbind(t18_1,t18_2,t19_1,t19_2,t20_1,t20_2,uow_consumption[3])
M3 <- cbind(t18_1,t18_2,t18_3,t19_1,t19_2,t19_3,t20_1,t20_2,t20_3,uow_consumption[3])
M4 <- cbind(t18_1,t18_2,t18_3,t18_4,t19_1,t19_2,t19_3,t19_4,t20_1,t20_2,t20_3,t20_4,uow_consumption[3])
M5 <- cbind(t18_1,t18_2,t18_3,t18_4,t18_7,t19_1,t19_2,t19_3,t19_4,t19_7,t20_1,t20_2,t20_3,t20_4,t20_7,uow_consumption[3])



# Rename column names in I/O Matrixes
colnames(M1)<-c("t18_1","t19_1","t20_1","output")
colnames(M2)<-c("t18_1","t18_2","t19_1","t19_2","t20_1","t20_2","output")
colnames(M3)<-c("t18_1","t18_2","t18_3","t19_1","t19_2","t19_3","t20_1","t20_2",'t20_3',"output")
colnames(M4)<-c("t18_1","t18_2","t18_3","t18_4","t19_1","t19_2","t19_3","t19_4","t20_1","t20_2","t20_3","t20_4","output")
colnames(M5)<-c("t18_1","t18_2","t18_3","t18_4","t18_7","t19_1","t19_2","t19_3","t19_4","t19_7","t20_1","t20_2","t20_3","t20_4","t20_7","output")



# Remove missing values
M1 <- M1[complete.cases(M1),]
M2 <- M2[complete.cases(M2),]
M3 <- M3[complete.cases(M3),]
M4 <- M4[complete.cases(M4),]
M5 <- M5[complete.cases(M5),]




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
model1 <- neuralnet(output ~ t18_1+t19_1+t20_1, data = M1_train_norm, hidden = 6, act.fct = 'logistic', linear.output = T)
plot(model1)
model1_results <- neuralnet::compute(model1,M1_test_norm[1:3])
predicted_output_model1 <- unnormalize(model1_results$net.result,min_output_M1,max_output_M1) #obtain predicted output
evaluate(original_test_outputs_M1,predicted_output_model1)  # evaluation



#M2
original_train_outputs_M2 <- M2[1:380,"output"]
original_test_outputs_M2 <- M2[381:468,"output"]

min_output_M2 <- min(original_train_outputs_M2)
max_output_M2 <- max(original_train_outputs_M2)

# neural network code/training
model2 <- neuralnet(output ~ t18_1+t18_2+t19_1+t19_2+t20_1+t20_2, data = M2_train_norm, hidden = c(8,6), act.fct = 'logistic', linear.output = T)
plot(model2)
model2_results <- neuralnet::compute(model2,M2_test_norm[1:6])
predicted_output_model2 <- unnormalize(model2_results$net.result,min_output_M2,max_output_M2) #obtain predicted output
evaluate(original_test_outputs_M2,predicted_output_model2)  # evaluation



#M3
original_train_outputs_M3 <- M3[1:380,"output"]
original_test_outputs_M3 <- M3[381:467,"output"]

min_output_M3 <- min(original_train_outputs_M3)
max_output_M3 <- max(original_train_outputs_M3)


# neural network code/training
model3 <- neuralnet(output ~ t18_1+t18_2+t18_3+t19_1+t19_2+t19_3+t20_1+t20_2+t20_3,data = M3_train_norm, hidden = c(10,6), act.fct = 'logistic', linear.output = T)
plot(model3)
model3_results <- neuralnet::compute(model3,M3_test_norm[1:9])
predicted_output_model3 <- unnormalize(model3_results$net.result,min_output_M3,max_output_M3) #obtain predicted output
evaluate(original_test_outputs_M3,predicted_output_model3)  # evaluation



#M4
original_train_outputs_M4 <- M4[1:380,"output"]
original_test_outputs_M4 <- M4[381:466,"output"]

min_output_M4 <- min(original_train_outputs_M4)
max_output_M4 <- max(original_train_outputs_M4)

# neural network code/training
model4 <- neuralnet(output ~ t18_1+t18_2+t18_3+t18_4+t19_1+t19_2+t19_3+t19_4+t20_1+t20_2+t20_3+t20_4, data = M4_train_norm, hidden = c(8,4), act.fct = 'tanh', linear.output = T)
plot(model4)
model4_results <- neuralnet::compute(model4,M4_test_norm[1:12])
predicted_output_model4 <- unnormalize(model4_results$net.result,min_output_M4,max_output_M4) #obtain predicted output
evaluate(original_test_outputs_M4,predicted_output_model4)  # evaluation



#M5
original_train_outputs_M5 <- M5[1:380,"output"]
original_test_outputs_M5 <- M5[381:463,"output"]

min_output_M5 <- min(original_train_outputs_M5)
max_output_M5 <- max(original_train_outputs_M5)

# neural network code/training
model5 <- neuralnet(output ~ t18_1+t18_2+t18_3+t18_4+t18_7+t19_1+t19_2+t19_3+t19_4+t19_7+t20_1+t20_2+t20_3+t20_4+t20_7, data = M5_train_norm, hidden = c(10,6), act.fct = 'logistic', linear.output = T)
plot(model5)
model5_results <- neuralnet::compute(model5,M5_test_norm[1:15])
predicted_output_model5 <- unnormalize(model5_results$net.result,min_output_M5,max_output_M5) #obtain predicted output
evaluate(original_test_outputs_M5,predicted_output_model5)  # evaluation
