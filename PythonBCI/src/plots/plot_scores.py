import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os

def plot_scores():
    # LaTeX/Overleaf friendly font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    csv_path = r'.\PythonBCI\data\processed\scores.csv'
    df = pd.read_csv(csv_path)

    rounds = [1, 2, 3]

    plt.figure(figsize=(8, 3))
    
    num_users = len(df[df['id'] != 'gold'])
    orange_colors = cm.Oranges(np.linspace(0.4, 0.7, max(num_users, 1)))
    
    player_num = 1
    user_idx = 0
    for idx, row in df.iterrows():
        raw_id = row['id']
        scores = [row['game_1'], row['game_2'], row['game_3']]
        if raw_id == 'gold':
            player_label = 'simulation'
            color = '#B41FFF'  # Purple
        else:
            player_label = "user " + str(player_num)
            player_num += 1
            color = orange_colors[user_idx]
            user_idx += 1
            
        plt.plot(rounds, scores, label=player_label, color=color, linewidth=2)
    
    plt.title('Scores Trend over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.ylim(0, 12)
    
    # Ensure x-axis only shows the integer rounds
    plt.xticks(rounds)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(title='Player', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    output_filename = 'scores_trend.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved successfully to {os.path.abspath(output_filename)}")

if __name__ == '__main__':
    plot_scores()
