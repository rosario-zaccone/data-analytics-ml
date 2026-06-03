import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Test R2:", r2_score(y_test, y_pred))
    print("Test MSE:", mean_squared_error(y_test, y_pred))
    print("Test RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("Test MAE:", mean_absolute_error(y_test, y_pred))

    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.7)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--")
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title("Actual vs Predicted Values")
    axes[0].grid(True)

    # 2. Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.7)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted Values")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    print()