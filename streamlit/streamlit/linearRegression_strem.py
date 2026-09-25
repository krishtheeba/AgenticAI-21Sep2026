import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Linear Regression Demo")

# Sidebar inputs
st.sidebar.header("Data Settings")
n_points = st.sidebar.slider("Number of data points", 10, 200, 50)
noise_level = st.sidebar.slider("Noise level", 0.0, 10.0, 2.0)

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, n_points)
y = 3 * X + 7 + np.random.randn(n_points) * noise_level

# Reshape X for sklearn
X_reshaped = X.reshape(-1, 1)

# Fit linear regression
model = LinearRegression()
model.fit(X_reshaped, y)

# Predict
y_pred = model.predict(X_reshaped)

# Show model parameters
st.write(f"**Slope (Coefficient):** {model.coef_[0]:.2f}")
st.write(f"**Intercept:** {model.intercept_:.2f}")

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, label="Data", color="blue")
ax.plot(X, y_pred, color="red", label="Regression Line")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)

# Show data table
df = pd.DataFrame({"X": X, "y": y, "Predicted y": y_pred})
st.dataframe(df)