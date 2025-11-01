"""
Generate all visualization graphs for Linear Regression teaching materials
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
data = pd.read_csv('linear_regression_sample.csv')
X = data[['Hours_Studied']]
y = data['Exam_Score']

# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
m = model.coef_[0]
b = model.intercept_

print("Generating graphs...")

# ============================================================================
# Graph 1: Basic Scatter Plot with Regression Line
# ============================================================================
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=150, alpha=0.7, 
            label='Actual Students', edgecolors='black', linewidth=2)
plt.plot(X, y_pred, color='red', linewidth=3, 
         label='Regression Line', linestyle='--')

plt.xlabel('Hours Studied', fontsize=14, fontweight='bold')
plt.ylabel('Exam Score', fontsize=14, fontweight='bold')
plt.title('Linear Regression: Study Time vs. Exam Score', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add equation
equation_text = f'y = {m:.2f}x + {b:.2f}\nR² = {r2:.4f}'
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=13, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('graph1_regression_line.png', dpi=300, bbox_inches='tight')
print("✓ Saved: graph1_regression_line.png")
plt.close()

# ============================================================================
# Graph 2: Residual Plot
# ============================================================================
residuals = y - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple', s=150, alpha=0.7, 
            edgecolors='black', linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', linewidth=3, label='Zero Error Line')

plt.xlabel('Predicted Score', fontsize=14, fontweight='bold')
plt.ylabel('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
plt.title('Residual Plot: Checking Model Errors', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add interpretation
note_text = 'Points close to zero line\n= Good predictions'
plt.text(0.05, 0.95, note_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('graph2_residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: graph2_residual_plot.png")
plt.close()

# ============================================================================
# Graph 3: Scatter Plot Only (Raw Data)
# ============================================================================
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='green', s=200, alpha=0.6, 
            edgecolors='black', linewidth=2)

plt.xlabel('Hours Studied', fontsize=14, fontweight='bold')
plt.ylabel('Exam Score', fontsize=14, fontweight='bold')
plt.title('Raw Data: Study Hours vs Exam Scores', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add annotations for a few points
plt.annotate('2 hrs → 55', xy=(2, 55), xytext=(3, 50),
             fontsize=11, ha='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.annotate('11 hrs → 95', xy=(11, 95), xytext=(9, 90),
             fontsize=11, ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

plt.tight_layout()
plt.savefig('graph3_raw_data.png', dpi=300, bbox_inches='tight')
print("✓ Saved: graph3_raw_data.png")
plt.close()

# ============================================================================
# Graph 4: Prediction Comparison (Actual vs Predicted)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(data))
width = 0.35

bars1 = ax.bar(x_pos - width/2, y, width, label='Actual Score',
               color='skyblue', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, y_pred, width, label='Predicted Score',
               color='salmon', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Student (by Hours Studied)', fontsize=14, fontweight='bold')
ax.set_ylabel('Exam Score', fontsize=14, fontweight='bold')
ax.set_title('Actual vs Predicted Scores', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{h}h' for h in X.values.flatten()], fontsize=10)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('graph4_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: graph4_actual_vs_predicted.png")
plt.close()

# ============================================================================
# Graph 5: Combined Dashboard (2x2 Grid)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Scatter with line
axes[0, 0].scatter(X, y, color='blue', s=100, alpha=0.6, edgecolors='black')
axes[0, 0].plot(X, y_pred, color='red', linewidth=2, linestyle='--')
axes[0, 0].set_xlabel('Hours Studied', fontweight='bold')
axes[0, 0].set_ylabel('Exam Score', fontweight='bold')
axes[0, 0].set_title('Regression Line', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Top-right: Residuals
axes[0, 1].scatter(y_pred, residuals, color='purple', s=100, alpha=0.6, edgecolors='black')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Score', fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontweight='bold')
axes[0, 1].set_title('Residual Plot', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Bottom-left: Bar comparison
x_pos = np.arange(len(data))
axes[1, 0].bar(x_pos, y, alpha=0.7, label='Actual', color='skyblue', edgecolor='black')
axes[1, 0].plot(x_pos, y_pred, color='red', marker='o', linewidth=2, 
                markersize=8, label='Predicted')
axes[1, 0].set_xlabel('Student Index', fontweight='bold')
axes[1, 0].set_ylabel('Score', fontweight='bold')
axes[1, 0].set_title('Comparison', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Bottom-right: Metrics summary (text)
axes[1, 1].axis('off')
metrics_text = f"""
MODEL PERFORMANCE

Formula: y = {m:.2f}x + {b:.2f}

Slope (m): {m:.2f}
→ Each hour adds {m:.2f} points

Intercept (b): {b:.2f}
→ Base score is {b:.2f} points

R² Score: {r2:.4f} ({r2*100:.1f}%)
→ Model explains {r2*100:.1f}% of variation

RMSE: {rmse:.2f} points
→ Average error is ±{rmse:.2f} points
"""
axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Linear Regression Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('graph5_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: graph5_dashboard.png")
plt.close()

# ============================================================================
# Graph 6: Prediction Examples
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the regression line and data
ax.scatter(X, y, color='blue', s=150, alpha=0.6, 
           label='Actual Students', edgecolors='black', linewidth=2, zorder=3)
ax.plot(X, y_pred, color='red', linewidth=3, 
        label='Regression Line', linestyle='--', zorder=2)

# Add prediction examples
test_hours = [3.5, 6.5, 9.5]
colors = ['green', 'orange', 'purple']

for hours, color in zip(test_hours, colors):
    pred_score = model.predict([[hours]])[0]
    ax.scatter([hours], [pred_score], color=color, s=300, 
               marker='*', edgecolors='black', linewidth=2,
               label=f'Predict: {hours}h → {pred_score:.1f}', zorder=4)
    ax.vlines(hours, 0, pred_score, colors=color, linestyles='dotted', linewidth=2, alpha=0.7)
    ax.hlines(pred_score, 0, hours, colors=color, linestyles='dotted', linewidth=2, alpha=0.7)

ax.set_xlabel('Hours Studied', fontsize=14, fontweight='bold')
ax.set_ylabel('Exam Score', fontsize=14, fontweight='bold')
ax.set_title('Making Predictions with Linear Regression', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 12)
ax.set_ylim(40, 100)

plt.tight_layout()
plt.savefig('graph6_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: graph6_predictions.png")
plt.close()

print("\n" + "="*50)
print("✅ All graphs generated successfully!")
print("="*50)
print("\nGenerated files:")
print("  1. graph1_regression_line.png - Basic regression visualization")
print("  2. graph2_residual_plot.png - Error analysis")
print("  3. graph3_raw_data.png - Original data points")
print("  4. graph4_actual_vs_predicted.png - Bar comparison")
print("  5. graph5_dashboard.png - Complete analysis dashboard")
print("  6. graph6_predictions.png - Prediction examples")
print("\nYou can use these in presentations, documents, or share with students!")
