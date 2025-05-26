import numpy as np
import matplotlib.pyplot as plt

# Functions and their derivatives
def x2(x):
    return x * x

def x2_(x):
    return 2 * x

def x4(x):
    return x ** 4

def x4_(x):
    # Use np.copysign to prevent overflow when x is very large
    return 4 * (np.copysign(1, x) * min(abs(x) ** 3, 1e10))

def sin_x(x):
    return np.sin(x)

def sin_x_(x):
    return np.cos(x)

# Momentum update
def momentum_update(velocity, gradient, momentum=0.9):
    return momentum * velocity - gradient

# Nesterov update
def nesterov_update(x, velocity, gradient_func, lr, momentum=0.9):
    lookahead = x + momentum * velocity
    gradient = gradient_func(lookahead)
    velocity = momentum * velocity - lr * gradient
    return velocity

# AdaGrad update
def adagrad_update(gradient, historical_gradient, epsilon=1e-8):
    historical_gradient += gradient ** 2
    adjusted_lr = lr / (np.sqrt(historical_gradient) + epsilon)
    return -adjusted_lr * gradient, historical_gradient

# Training loop for different methods
def train(method, gradient_func, x_start, num_steps, lr, momentum=0.9):
    global update
    x = x_start
    velocity = 0
    historical_gradient = 0
    history = []

    for step in range(num_steps):
        gradient = gradient_func(x)

        if method == "sgd":
            update = -lr * gradient

        elif method == "momentum":
            velocity = momentum_update(velocity, gradient, momentum)
            update = velocity

        elif method == "nesterov":
            velocity = nesterov_update(x, velocity, gradient_func, lr, momentum)
            update = velocity

        elif method == "adagrad":
            update, historical_gradient = adagrad_update(gradient, historical_gradient)

        x += update
        history.append(x)

    return history

# Parameters
x_start = 10.0
lr = 0.01
num_steps = 200  # Increased for more clarity in convergence
momentum = 0.9

# Train models
history = {
    'sgd_x2': train("sgd", x2_, x_start, num_steps, lr),
    'sgd_x4': train("sgd", x4_, x_start, num_steps, lr),
    'momentum_x2': train("momentum", x2_, x_start, num_steps, lr, momentum),
    'momentum_x4': train("momentum", x4_, x_start, num_steps, lr, momentum),
    'nag_x2': train("nesterov", x2_, x_start, num_steps, lr, momentum),
    'nag_x4': train("nesterov", x4_, x_start, num_steps, lr, momentum),
    'adagrad_x2': train("adagrad", x2_, x_start, num_steps, lr),
    'adagrad_x4': train("adagrad", x4_, x_start, num_steps, lr),
    'sgd_sin': train("sgd", sin_x_, x_start, num_steps, lr),
    'momentum_sin': train("momentum", sin_x_, x_start, num_steps, lr, momentum),
    'nag_sin': train("nesterov", sin_x_, x_start, num_steps, lr, momentum),
    'adagrad_sin': train("adagrad", sin_x_, x_start, num_steps, lr),
}

# Visualization
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Plot x^2
for method in ['sgd_x2', 'momentum_x2', 'nag_x2', 'adagrad_x2']:
    axs[0].plot(history[method], label=method)
axs[0].set_title("Convergence of Optimization Methods on x^2")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Value of x")
axs[0].legend()
axs[0].grid()

# Plot x^4
for method in ['sgd_x4', 'momentum_x4', 'nag_x4', 'adagrad_x4']:
    axs[1].plot(history[method], label=method)
axs[1].set_title("Convergence of Optimization Methods on x^4")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Value of x")
axs[1].legend()
axs[1].grid()

# Plot sin(x)
for method in ['sgd_sin', 'momentum_sin', 'nag_sin', 'adagrad_sin']:
    axs[2].plot(history[method], label=method)
axs[2].set_title("Convergence of Optimization Methods on sin(x)")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Value of x")
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()

# Analysis
print("Final values:")
for method, values in history.items():
    print(f"{method}: {values[-1]:.6f}")

# Edge case tests
def test_edge_cases():
    edge_start_values = [1e6, -1e6, 1e-6, -1e-6]
    for start_value in edge_start_values:
        print(f"\nTesting with start value: {start_value}")
        for method in ['sgd', 'momentum', 'nesterov', 'adagrad']:
            edge_history = train(method, x4_, start_value, num_steps, lr, momentum)
            print(f"Final value for x4 with {method}: {edge_history[-1]:.6f}")

test_edge_cases()
