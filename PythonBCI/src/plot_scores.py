import matplotlib.pyplot as plt
import os

def plot_scores():
    rounds = [1, 2, 3]
    kaje_scores = [8, 2, 2]
    vikt_scores = [2, 3, 3]
    simulated_scores = [11, 6, 9]

    plt.figure(figsize=(8, 6))
    
    plt.plot(rounds, kaje_scores, marker='o', label='kaje', linewidth=2)
    plt.plot(rounds, vikt_scores, marker='s', label='vikt', linewidth=2)
    plt.plot(rounds, simulated_scores, marker='^', label='simulated', linewidth=2)
    
    plt.title('Scores Trend over Rounds', fontsize=16)
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    
    # Ensure x-axis only shows the integer rounds
    plt.xticks(rounds)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(title='Player / Type', fontsize=12)
    plt.tight_layout()
    
    output_filename = 'scores_trend.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {os.path.abspath(output_filename)}")

if __name__ == '__main__':
    plot_scores()
