import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LOG_FILE = 'training_log.jsonl'
SMOOTHING_WINDOW = 100
Y_MAX_LIMIT = 2.5

def parse_logs(filename):
    train_data = []
    val_data = []

    with open(filename, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'val_loss' in entry:
                    val_data.append(entry)
                elif 'train_loss' in entry:
                    train_data.append(entry)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(train_data), pd.DataFrame(val_data)

df_train, df_val = parse_logs(LOG_FILE)

df_train['smooth_loss'] = df_train['train_loss'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.despine()

plt.plot(df_train['step'], df_train['smooth_loss'],
         color='#0052cc', linewidth=2, label='Training Loss')

if not df_val.empty:
    plt.plot(df_val['step'], df_val['val_loss'],
             color='#d62728', linewidth=2, linestyle='-', label='Validation Loss')

plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
plt.ylabel('Cross Entropy Loss', fontsize=12, fontweight='bold')
plt.title('Best Model Convergence (206M Parameters)', fontsize=14, pad=15)
plt.legend(frameon=True, fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

plt.ylim(0.4, Y_MAX_LIMIT)
plt.xlim(0, df_train['step'].max())

plt.tight_layout()
plt.savefig('best_model_loss_clean.png', dpi=300)
plt.show()

print(f"Plot saved. Final Val Loss: {df_val['val_loss'].iloc[-1]:.4f}")
