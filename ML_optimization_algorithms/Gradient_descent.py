# 📦 Линейная регрессия с нуля на NumPy (уникальный стиль)
# Обучение модели y = wx + b вручную, без библиотек ML

import numpy as np
import matplotlib.pyplot as plt

# 1. Генерация данных
np.random.seed(42)
X_data = np.random.randn(10, 1)
Y_target = 5 * X_data + np.random.randn(10, 1) 

# 2. Инициализация параметров
weight = 0.0
bias = 0.0
lr = 0.1
epochs = 200
losses = []
#
# 3. Обучение модели вручную
for epoch in range(epochs):
    grad_w = 0.0
    grad_b = 0.0
    N = X_data.shape[0]

    for x_i, y_i in zip(X_data, Y_target):
        y_hat = weight * x_i + bias
        error = y_i - y_hat
        grad_w += -2 * x_i * error
        grad_b += -2 * error

    weight -= lr * grad_w / N
    bias  -= lr * grad_b / N

    # Вычисление ошибки (MSE)
    y_pred = weight * X_data + bias
    loss = np.mean((Y_target - y_pred)**2)
    losses.append(loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={weight.item():.2f}, b={bias.item():.2f}")

# 4. Визуализация результата обучения
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.scatter(X_data, Y_target, label='Данные')
plt.plot(X_data, y_pred, color='red', label='Предсказание')
plt.title('Линейная регрессия')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses, label='Loss')
plt.title('Убывание ошибки (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
