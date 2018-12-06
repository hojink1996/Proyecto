"""
Script that Graphs the Learning Curves of the retraining of the model.

Authors: Hojin Kang and Tomas Nunez
"""

import json
import matplotlib.pyplot as plt

# Open the JSON files
with open('data.json', 'r') as f:
    dict = json.load(f)
with open('data2.json', 'r') as f:
    dict2 = json.load(f)
with open('data3.json', 'r') as f:
    dict3 = json.load(f)
with open('data4.json', 'r') as f:
    dict4 = json.load(f)
with open('data5.json', 'r') as f:
    dict5 = json.load(f)

# Get Accuracy
val_acc = dict['val_acc']
train_acc = dict['acc']
val_acc2 = dict2['val_acc']
train_acc2 = dict2['acc']
val_acc3 = dict3['val_acc']
train_acc3 = dict3['acc']
val_acc4 = dict4['val_acc']
train_acc4 = dict4['acc']
val_acc5 = dict5['val_acc']
train_acc5 = dict5['acc']

# Get the x axis
x_vals = range(len(val_acc))
x_vals2 = range(len(val_acc2))
x_vals3 = range(len(val_acc3))
x_vals4 = range(len(val_acc4))
x_vals5 = range(len(val_acc5))

# Get the maximum value
m = max(val_acc)
max_pos = [i for i, j in enumerate(val_acc) if abs(j - m) < 0.001]
m2 = max(val_acc2)
max_pos2 = [i for i, j in enumerate(val_acc2) if abs(j - m2) < 0.0001]
m3 = max(val_acc3)
max_pos3 = [i for i, j in enumerate(val_acc3) if abs(j - m3) < 0.0001]
m4 = max(val_acc4)
max_pos4 = [i for i, j in enumerate(val_acc4) if abs(j - m4) < 0.0001]
m5 = max(val_acc5)
max_pos5 = [i for i, j in enumerate(val_acc5) if abs(j - m5) < 0.0001]

# Plot
plt.figure(figsize=(12,8))
plt.plot(max_pos, m, 'r*', label='Best Model (Chosen) - Modelo 1')
plt.plot(x_vals, val_acc, label='Validation Accuracy - Modelo 1')
plt.plot(x_vals, train_acc, label='Training Accuracy - Modelo 1')
plt.plot(max_pos2, m2, 'r*', label='Best Model (Chosen) - Modelo 2')
plt.plot(x_vals2, val_acc2, label='Validation Accuracy - Modelo 2')
plt.plot(x_vals2, train_acc2, label='Training Accuracy - Modelo 2')
plt.plot(max_pos3, m3, 'r*', label='Best Model (Chosen) - Modelo 3')
plt.plot(x_vals3, val_acc3, label='Validation Accuracy - Modelo 3')
plt.plot(x_vals3, train_acc3, label='Training Accuracy - Modelo 3')
plt.plot(x_vals4, val_acc4, label='Validation Accuracy - Modelo 4')
plt.plot(x_vals4, train_acc4, label='Training Accuracy - Modelo 4')
plt.plot(max_pos4, m4, 'r*', label='Best Model (Chosen) - Modelo 4')
plt.plot(x_vals5, val_acc5, label='Validation Accuracy - Modelo 5')
plt.plot(x_vals5, train_acc5, label='Training Accuracy - Modelo 5')
plt.plot(max_pos4, m5, 'r*', label='Best Model (Chosen) - Modelo 5')

# Give the plot parameters
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.title('Fine tuning the model', fontsize=14)
plt.legend(loc=1)
plt.grid()

# Show the plot
plt.savefig('learning_curve_total.png', dpi=300)
plt.show()

# Print the maximum accuracies
print(str(m) + ' ' + str(m2) + ' ' + str(m3) + ' ' + str(m4) + ' ' + str(m5))
