import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


transformers = load_json('scaling_results.json') + load_json('scaling_results_xl.json')
lstms = load_json('rnn_results.json') + load_json('rnn_results_xl.json')


transformers.sort(key=lambda x: x['params'])
lstms.sort(key=lambda x: x['params'])


t_params = [x['params'] for x in transformers]
t_loss = [x['val_loss'] for x in transformers]
l_params = [x['params'] for x in lstms]
l_loss = [x['val_loss'] for x in lstms]


plt.figure(figsize=(10, 6))
plt.loglog(t_params, t_loss, marker='o', linestyle='-', linewidth=2, color='#1f77b4', label='Transformer (GPT-2)')
plt.loglog(l_params, l_loss, marker='s', linestyle='--', linewidth=2, color='#ff7f0e', label='LSTM Baseline')


log_N = np.log(t_params)
log_L = np.log(t_loss)
coeffs = np.polyfit(log_N, log_L, 1)
alpha = -coeffs[0]
plt.text(t_params[-2], t_loss[-2] + 0.05, f'Scaling Exponent $\\alpha \\approx {alpha:.3f}$', fontsize=12)

plt.xlabel('Parameters (N)', fontsize=12)
plt.ylabel('Validation Loss (Cross Entropy)', fontsize=12)
plt.title('Neural Scaling Laws: Symbolic Music Modeling', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(fontsize=12)
plt.savefig('scaling_plot.png', dpi=300)
plt.show()



t_time = [x['duration_sec'] for x in transformers]
l_time = [x['duration_sec'] for x in lstms]

plt.figure(figsize=(10, 6))
plt.scatter(t_time, t_loss, color='#1f77b4', s=100, label='Transformer')
plt.scatter(l_time, l_loss, color='#ff7f0e', marker='s', s=100, label='LSTM')


for i, txt in enumerate(['Tiny', 'Small', 'Med', 'Lrg', 'XL']):
    plt.annotate(txt, (t_time[i], t_loss[i]), xytext=(5, 5), textcoords='offset points')
    plt.annotate(txt, (l_time[i], l_loss[i]), xytext=(5, 5), textcoords='offset points')

plt.xlabel('Training Time (seconds)', fontsize=12)
plt.ylabel('Final Validation Loss', fontsize=12)
plt.title('Compute Efficiency: Time to Convergence (1 Epoch)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('efficiency_plot.png', dpi=300)
plt.show()
