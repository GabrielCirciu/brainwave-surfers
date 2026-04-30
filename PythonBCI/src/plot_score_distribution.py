import matplotlib.pyplot as plt
import numpy as np
import math
import os

def negative_binomial_pmf(k, r, p):
    """
    Calculates the probability of getting exactly k successes before r failures.
    k: number of correct choices (score)
    r: number of mistakes allowed before game over (lives)
    p: accuracy (probability of a correct choice)
    """
    return math.comb(k + r - 1, r - 1) * (p**k) * ((1-p)**r)

def plot_distributions():
    accuracies = [0.90, 0.80, 0.55]
    lives = 3
    
    # Evaluate up to a score of 40 rounds as requested
    k_values = np.arange(0, 41)
    
    plt.figure(figsize=(8, 6))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    
    for i, p in enumerate(accuracies):
        pmf = [negative_binomial_pmf(k, lives, p) for k in k_values]
        
        # Theoretical expected score: E[k] = r * p / (1 - p)
        expected_score = lives * p / (1 - p)
        
        label = f'Accuracy {int(p*100)}% (Expected Score: {expected_score:.1f})'
        plt.plot(k_values, pmf, linestyle='-', color=colors[i], label=label, linewidth=2)
        
        # Add a vertical dashed line at the expected score
        plt.axvline(x=expected_score, color=colors[i], linestyle='--', alpha=0.7)
        
    plt.title(f'Score Distribution based on Accuracy (Game over after {lives} mistakes)', fontsize=16)
    plt.xlabel('Score (Number of Correct Choices before 3 Mistakes)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    output_filename = 'score_distribution.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {os.path.abspath(output_filename)}")
    
    print("\n--- Expected Scores ---")
    for p in accuracies:
        exp_score = lives * p / (1 - p)
        print(f"Accuracy {int(p*100)}%: Expected Score = {exp_score:.2f}")

if __name__ == '__main__':
    plot_distributions()
