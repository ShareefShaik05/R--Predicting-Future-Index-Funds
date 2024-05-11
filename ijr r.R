setwd("C:/Users/share/OneDrive/Documents/Shareef DMCA project dataset")

data = read.csv("IJR (1).csv")


data$date = paste(data$Year, data$Month, sep = "/")

dmdata = data[c("date", "Price")]

dmdata

#Predicting IRJ Index Price
IJR.ts = ts(dmdata$Price, start = c(2000, 1), end = c(2022, 12), freq = 12)
summary(IJR.ts)

length(IJR.ts)

ylow = round(summary(IJR.ts)[1],-4)
yhigh = summary(IJR.ts)[6] + 13
paste(ylow, yhigh)

options(scipen = 999) # prevent scientific notation from showing on the plot

plot(IJR.ts, ylim = c(ylow, yhigh),  ylab = "iShares Core S&P Small-Cap ETF (IJR) ", 
     xlab = "Year", bty = "l", xaxt = "n", xlim = c(2000, (2022)), main = "iShares Core S&P Small-Cap ETF (IJR)")
axis(1, at = seq(2000, 2022, 1), labels = format(seq(2000, 2022, 1)))

# The time-series components in evidence are level, trend, seasonality, and noise.

#partitioning

nValid = 12
nTrain = length(IJR.ts) - nValid

trainIJR.ts = window(IJR.ts, start = c(2000, 1), end = c(2021, 12))
validIJR.ts = window(IJR.ts, start = c(2022, 1), end = c(2022, 12))

#model building

#ETS
set.seed(500)

library(forecast)
IJRETS = ets(trainIJR.ts, model = "ZZZ") 
summary(IJRETS)

IJRETS.pred = forecast(IJRETS, h = nValid, level = 0)
accuracy(IJRETS.pred, validIJR.ts) # 
IJRETS.pred$method
IJRETS.pred$model$aic 

plot(IJRETS.pred, ylim = c(ylow, yhigh),  ylab = "iShares Core S&P Small-Cap ETF (IJR)", 
     xlab = "date", bty = "l", xaxt = "n", xlim = c(2000, (2023)), main = "iShares Core S&P Small-Cap ETF (IJR) - Actual Vs Forecast")
axis(1, at = seq(2000, 2023, 1), labels = format(seq(2000, 2023, 1)))
lines(validIJR.ts, lwd = 2, col = "darkorange") 
lines(IJRETS.pred$fitted, lwd = 1.5, col = "blue", lty = 5) 

IJRETS.pred


#ARIMA
# Load the forecast package
library(forecast)


# Fit ARIMA model
arima_model <- auto.arima(trainIJR.ts)

# Print ARIMA model summary
summary(arima_model)

# Forecast using ARIMA model
forecasted_values <- forecast(arima_model, h = nValid)

# Print forecasted values
print(forecasted_values)

# Plot the actual and forecasted values
plot(forecasted_values, main = "ARIMA Forecast for iShares Core S&P Small-Cap ETF (IJR)", xlab = "Year", ylab = "iShares Core S&P Small-Cap ETF (IJR) - Actual Vs Forecast", ylim = c(ylow, yhigh), 
     bty = "l", xaxt = "n", xlim = c(2000, 2022))
axis(1, at = seq(2000, 2022, 1), labels = format(seq(2000, 2022, 1)))

accuracy_AR <- accuracy(forecasted_values, validIJR.ts)
print(accuracy_AR)



#Neuro networks
IJR.nnetar = nnetar(trainIJR.ts, P = 12, scale.inputs = TRUE )

IJR.nnetar$fitted
IJR.nnetar.pred <- forecast( IJR.nnetar, h=nValid )
accuracy(IJR.nnetar.pred, validIJR.ts)

plot(trainIJR.ts, ylim = c(ylow, yhigh),  ylab = "iShares Core S&P Small-Cap ETF (IJR)", 
     xlab = "date", bty = "l", xaxt = "n", xlim = c(2000, (2022)), main = "iShares Core S&P Small-Cap ETF (IJR) - Actual vs Forecast")
axis(1, at = seq(2000, 2022, 1), labels = format(seq(2000, 2022, 1)))

lines( IJR.nnetar.pred$fitted, lwd = 2, col = "blue", lty = 1 ) 
lines( IJR.nnetar.pred$mean, lwd = 2, col = "darkorange", lty = 1 ) 
lines( validIJR.ts, col="purple", lwd=2 )

lines( c(2021, 2021), c(ylow, yhigh) )
lines( c(2022, 2022), c(ylow, yhigh) )
text(2020.5, yhigh, "T") # Training
text(2021.5, yhigh, "V") # Validation
text(2022.5, yhigh, "F") # Future

legend("topleft", legend=c("Actual Training","Fitted Training", 
                           "Predicted Validation", "Actual Validation"),
       col=c("green", "blue", "darkorange", "purple"), lwd=2, cex=1.0)




###

