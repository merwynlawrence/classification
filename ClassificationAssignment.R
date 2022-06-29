library(dplyr)
library(janitor)
library(skimr)
library(party)
library(data.table)
library (caret)

#read the csv file  into Rstudio
diab <- read.csv("diabetes.csv", header = TRUE)

#inspect the data
names(diab)

#   > names(diab)
# [1] "Pregnancies"              "Glucose"                  "BloodPressure"           
# [4] "SkinThickness"            "Insulin"                  "BMI"                     
# [7] "DiabetesPedigreeFunction" "Age"                      "Outcome"    
head (diab)

tail(diab)

summary((diab))

str(diab)


#checking the dimensions
nrow(diab)
ncol(diab)
dim(diab)

# #checking the dimensions
# > nrow(diab)
# [1] 768
# > ncol(diab)
# [1] 9
# > dim(diab)
# [1] 768   9

# 
# Since we need categorical (Factor) data to class variable for prediction, 
#hence we should convert the NSP variable to categorical form 
#by running the following command.

#converting Outcome to categorical form
diab$Outcome <- as.factor(diab$Outcome)


str(diab)

head(diab)

#deleting unwanted COlums of pregnancies and skin thinckness and Insulin

df <- diab[, c(2,3,6,7,8,9)]
summary(df)
dim(df)
names(df)
tail(df)
ncol(df)

#cleaning DataSet
skim(df)


setDT(df)[BloodPressure==0, BloodPressure := NA, ]
setDT(df)[Glucose==0, Glucose := NA, ]
setDT(df)[BMI==0, BMI := NA, ]


df<- df%>% mutate(BloodPressure = ifelse(is.na(BloodPressure), mean(BloodPressure, na.rm=T),BloodPressure))
df<- df%>% mutate(Glucose = ifelse(is.na(Glucose), mean(Glucose, na.rm=T),Glucose))
df<- df%>% mutate(BMI = ifelse(is.na(BMI), mean(BMI, na.rm=T),BMI))

summary(df)


# box plot is now taken of the variables to check for outliers
boxplot(df$BloodPressure)
boxplot(df$Glucose)
boxplot(df$BMI)

dim(df)


#df[, BloodPressure := replace(BloodPressure, is.na(BloodPressure), mean(BloodPressure, na.rm = T)), by = .1(mean(df$BloodPressure))]
df
print(df)

outlier_values <- boxplot.stats(df$BloodPressure)$out
outlier_indices <- which(df$BloodPressure %in% outlier_values)
df <- df[-outlier_indices,]
boxplot(df$BloodPressure)
df

outlier_valuesBMI <- boxplot.stats(df$BMI)$out
outlier_indicesBMI <- which((df$BMI %in% outlier_valuesBMI))
df <- df[-outlier_indicesBMI, ]
boxplot(df$BMI)

summary(df)
view(df)

# save the dataset for further use using the code:
write.csv(df,"diabetes_clean.csv", quote = FALSE, row.names = FALSE)


#training and Validation
set.seed(1234)
pd <- sample(2, nrow(df), replace = TRUE, prob = c(0.8,0.2)) #Dividing the dataset into training and validation 

train <- df[pd==1,]
test <- df[pd==2,]

dim(train)
dim(test)
print(df$Outcome)

#training the tree using ctree function 
# Building the classification model for the Target variable Outcome 
#using the other variables
df_tree <- ctree(Outcome ~ Glucose + BloodPressure + BMI + DiabetesPedigreeFunction + Age, data=train)
df_tree

print(df_tree)

#Plotting the results as a decision tree  using
plot(df_tree)
plot( df_tree, type="simple")


#check the prediction on train data
predict(df_tree)
train$Outcome

#Generate Frequency tables: 
tab <- table(predict(df_tree), train$Outcome)


print(tab)

#caluculate Accuracy and Error
sum(diag(tab))/sum(tab)
1-sum(diag(tab))/sum(tab)


# > sum(diag(tab))/sum(tab)
# [1] 0.7894737
# > 1-sum(diag(tab))/sum(tab)
# [1] 0.2105263

#the above result shows us that Classification is 79% accurate
#and classification error is 21%

#We now check the validation data performance
test_predict <- table(predict(df_tree, newdata= test), test$Outcome )
print(test_predict)


#Calculate accuracy

sum(diag(test_predict))/sum(test_predict)
1-sum(diag(test_predict))/sum(test_predict)

# > sum(diag(test_predict))/sum(test_predict)
# [1] 0.7974684
# > 1-sum(diag(test_predict))/sum(test_predict)
# [1] 0.2025316

#The validation results show 80% Accuracy and 20% classification error

##################################
##building a confusion matrix to  check the performance of the model


library(caret)
install.packages("e1071")
library(e1071)

confusionMatrix(data = test_predict, reference = test$Outcome)

# 
# 0  1
# 0 87 13
# 1 19 39

###Confusion matrix results show an accuracy of 80%
# True positive = 87, 
# true negative 39, 
# false positive 13, 
# false negative19
# sensitivity value of 0.82 means the model performed well in predicting the positive class
# specificity tells the model performance in predicting the negative class
# high specificity and sensitivity value means it is a good model
