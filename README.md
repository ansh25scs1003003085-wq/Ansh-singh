# ============================================
# AI-ML Project: Livestock Mortality Prediction
# Using Climate Data (Pure Python - No Libraries)
# ============================================

# Sample Dataset (can be replaced with real data)
temperature = [30, 32, 35, 40, 28]
humidity = [60, 65, 70, 80, 55]
rainfall = [200, 180, 150, 100, 220]
pasture = [80, 70, 60, 40, 85]

# Target: Mortality Risk (0 to 1)
mortality = [0.2, 0.3, 0.5, 0.8, 0.1]


# --------------------------------------------
# Function to calculate mean
def mean(values):
    return sum(values) / len(values)


# --------------------------------------------
# Train Linear Regression for ONE feature
def train_feature(x, y):
    x_mean = mean(x)
    y_mean = mean(y)

    num = 0
    den = 0

    for i in range(len(x)):
        num += (x[i] - x_mean) * (y[i] - y_mean)
        den += (x[i] - x_mean) ** 2

    weight = num / den
    bias = y_mean - weight * x_mean

    return weight, bias


# --------------------------------------------
# Train all features
w_temp, b_temp = train_feature(temperature, mortality)
w_hum, b_hum = train_feature(humidity, mortality)
w_rain, b_rain = train_feature(rainfall, mortality)
w_past, b_past = train_feature(pasture, mortality)


# --------------------------------------------
# Prediction Function (Multi-feature)
def predict(temp, hum, rain, past):

    pred1 = w_temp * temp + b_temp
    pred2 = w_hum * hum + b_hum
    pred3 = w_rain * rain + b_rain
    pred4 = w_past * past + b_past

    # Average prediction
    result = (pred1 + pred2 + pred3 + pred4) / 4

    # Clamp between 0 and 1
    if result < 0:
        result = 0
    if result > 1:
        result = 1

    return result


# --------------------------------------------
# Model Evaluation
def evaluate():
    total_error = 0

    print("\nModel Evaluation:\n")

    for i in range(len(temperature)):
        pred = predict(temperature[i], humidity[i], rainfall[i], pasture[i])
        actual = mortality[i]
        error = abs(pred - actual)

        total_error += error

        print(f"Sample {i+1}: Predicted={round(pred,2)} | Actual={actual} | Error={round(error,2)}")

    print("\nAverage Error:", round(total_error / len(temperature), 3))


# --------------------------------------------
# Show learned weights
def show_weights():
    print("\nModel Weights:")
    print("Temperature:", w_temp, b_temp)
    print("Humidity:", w_hum, b_hum)
    print("Rainfall:", w_rain, b_rain)
    print("Pasture:", w_past, b_past)


# --------------------------------------------
# Menu System
def menu():
    while True:
        print("\n====== AI Livestock Prediction System ======")
        print("1. Predict Mortality Risk")
        print("2. View Model Weights")
        print("3. Evaluate Model")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == '1':
            t = float(input("Temperature: "))
            h = float(input("Humidity: "))
            r = float(input("Rainfall: "))
            p = float(input("Pasture Quality: "))

            result = predict(t, h, r, p)
            print("\nPredicted Mortality Risk:", round(result, 3))

        elif choice == '2':
            show_weights()

        elif choice == '3':
            evaluate()

        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice!")


# --------------------------------------------
# Run Program
menu()
