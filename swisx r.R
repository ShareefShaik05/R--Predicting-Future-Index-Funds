setwd("C:/Users/share/OneDrive/Documents/Shareef DMCA project dataset")

data = read.csv("SWISX2 (1).csv")


data$date = paste(data$Year, data$Month, sep = "/")

dmdata = data[c("date", "Price")]

dmdata

#Predicting Schwab International Index Fund Index Price
swx.ts = ts(dmdata$Price, start = c(2000, 1), end = c(2022, 12), freq = 12)
summary(swx.ts)

length(swx.ts)

ylow = round(summary(swx.ts)[1],-4)
yhigh = summary(swx.ts)[6] + 13
paste(ylow, yhigh)

options(scipen = 999) # prevent scientific notation from showing on the plot

plot(swx.ts, ylim = c(ylow, yhigh),  ylab = "Schwab International Index Fund Index Price", 
     xlab = "Year", bty = "l", xaxt = "n", xlim = c(2000, (2022)), main = "Schwab International Index Fund Index Price - Plot")
axis(1, at = seq(2000, 2022, 1), labels = format(seq(2000, 2022, 1)))

# The time-series components in evidence are level, trend, seasonality, and noise.

#partitioning

nValid = 12
nTrain = length(swx.ts) - nValid

trainswx.ts = window(swx.ts, start = c(2000, 1), end = c(2021, 12))
validswx.ts = window(swx.ts, start = c(2022, 1), end = c(2022, 12))

#model building

#ETS
set.seed(500)
install.packages("forecast")

library(forecast)
swxETS = ets(trainswx.ts, model = "ZZZ") 
summary(swxETS)

swxETS.pred = forecast(swxETS, h = nValid, level = 0)
accuracy(swxETS.pred, validswx.ts) # 
swxETS.pred$method
swxETS.pred$model$aic 

plot(swxETS.pred, ylim = c(ylow, yhigh),  ylab = "Schwab International Index Fund", 
     xlab = "date", bty = "l", xaxt = "n", xlim = c(2000, (2023)), main = "Schwab International Index Fund - Actual Vs Forecast")
axis(1, at = seq(2000, 2023, 1), labels = format(seq(2000, 2023, 1)))
lines(validswx.ts, lwd = 2, col = "darkorange") 
lines(swxETS.pred$fitted, lwd = 1.5, col = "blue", lty = 5) 

swxETS.pred


#ARIMA
# Load the forecast package
library(forecast)


# Fit ARIMA model
arima_model <- auto.arima(trainswx.ts)

# Print ARIMA model summary
summary(arima_model)

# Forecast using ARIMA model
forecasted_values <- forecast(arima_model, h = nValid)

# Print forecasted values
print(forecasted_values)

# Plot the actual and forecasted values
plot(forecasted_values, main = "ARIMA Forecast for Schwab International Index Fund", xlab = "Year", ylab = "Schwab International Index Fund - Actual Vs Forecast", ylim = c(ylow, yhigh), 
     bty = "l", xaxt = "n", xlim = c(2000, 2022))
axis(1, at = seq(2000, 2022, 1), labels = format(seq(2000, 2022, 1)))

accuracy_AR <- accuracy(forecasted_values, validswx.ts)
print(accuracy_AR)



#Neuro networks
swx.nnetar = nnetar(trainswx.ts, P = 12, scale.inputs = TRUE )

swx.nnetar$fitted
swx.nnetar.pred <- forecast( swx.nnetar, h=nValid )
accuracy(swx.nnetar.pred, validswx.ts)

plot(trainswx.ts, ylim = c(ylow, yhigh),  ylab = "Schwab International Index Fund", 
     xlab = "date", bty = "l", xaxt = "n", xlim = c(2000, (2022)), main = "Schwab International Index Fund - Actual vs Forecast")
axis(1, at = seq(2000, 2022, 1), labels = format(seq(2000, 2022, 1)))

lines( swx.nnetar.pred$fitted, lwd = 2, col = "blue", lty = 1 ) 
lines( swx.nnetar.pred$mean, lwd = 2, col = "darkorange", lty = 1 ) 
lines( validswx.ts, col="purple", lwd=2 )

lines( c(2021, 2021), c(ylow, yhigh) )
lines( c(2022, 2022), c(ylow, yhigh) )
text(2020.5, yhigh, "T") # Training
text(2021.5, yhigh, "V") # Validation
text(2022.5, yhigh, "F") # Future

legend("topleft", legend=c("Actual Training","Fitted Training", 
                           "Predicted Validation", "Actual Validation"),
       col=c("green", "blue", "darkorange", "purple"), lwd=2, cex=1.0)




###

