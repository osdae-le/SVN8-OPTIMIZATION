 data <- read.csv("C://Users/18521/Downloads/PET_PRI_GND_DCUS_NUS_W.csv") 
 summary(data)
 library(forecast)

 data$Date <- as.Date(data$Date)

plot(data$Date, data$A1, type = "l", xlab = "Date", ylab = "A1", main = "Time Series Data")
ts_data_ts <- ts(data$A1)
arima_model <- auto.arima(ts_data_ts)
forecast_values <- forecast(arima_model, h = 52) # Forecasting 52 weeks ahead
plot(forecast_values, xlab = "Date", ylab = "A1 Forecast", main = "ARIMA Forecast for A1")
print(forecast_values)
write.csv(forecast_data, file = "forecasteds_values.csv", row.names = FALSE)
 

data$Date <- as.Date(data$Date)
set.seed(123)

train_indices <- sample(1:nrow(data), 0.8 * nrow(data), replace = FALSE)
test_indices <- setdiff(1:nrow(data), train_indices)

train_data <- data[train_indices, ]
test_data <- data[test_indices, ]

forecast_list <- list()

for (col in names(data)[-1]) {
ts_train_data <- ts(train_data[[col]])
arima_model <- auto.arima(ts_train_data)
forecast_values <- forecast(arima_model, h = nrow(test_data)) # Forecasting for the length of the testing set
forecast_list[[col]] <- forecast_values$mean
}

forecast_data <- data.frame(Date = test_data$Date, forecast_list)
library(Metrics)

 accuracy_metrics <- data.frame(Column = character(), MAE = numeric(), MSE = numeric(), RMSE = numeric(), stringsAsFactors = FALSE)

for (col in names(data)[-1]) {
actual_values <- test_data[[col]]  # Actual values from the test data
forecasted_values <- forecast_data[[col]]  # Forecasted values
mae <- mae(actual_values, forecasted_values)
mse <- mse(actual_values, forecasted_values)
rmse <- rmse(actual_values, forecasted_values)
accuracy_metrics <- rbind(accuracy_metrics, data.frame(Column = col, MAE = mae, MSE = mse, RMSE = rmse))
}
print(accuracy_metrics)
   Column       MAE       MSE      RMSE
1      A1 0.7461142 0.7733021 0.8793760
2      A2 0.7295301 0.7453000 0.8633076
3      A3 0.7767483 0.8220331 0.9066604
4      R1 0.7354208 0.7591008 0.8712639
5      R2 0.7195578 0.7322133 0.8556946
6      R3 0.7671421 0.8111537 0.9006407
7      M1 0.7762384 0.8105640 0.9003133
8      M2 0.7505473 0.7670319 0.8758036
9      M3 0.8079904 0.8657493 0.9304565
10     P1 0.8084106 0.8594340 0.9270566
11     P2 0.7903198 0.8269526 0.9093693
12     P3 0.8273441 0.8927301 0.9448440
13     D1 0.8590415 1.0314781 1.0156171

