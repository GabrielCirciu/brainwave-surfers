using UnityEngine;
using TMPro;

public class LifeManager : MonoBehaviour
{
    private int lives = 3;
    public TextMeshProUGUI livesText;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void LoseLife()
    {
        lives--;
        Debug.Log("Lost a life! Remaining lives: " + lives);
        livesText.text = "Lifes: " + lives;
        if (lives <= 0)
        {
            lives = 0;
            Debug.Log("Game Over!");
            ScoreManager.Instance.EndGame();
        }
    }

    public void Reset() 
    {
        lives = 3;
        livesText.text = "Lifes: " + lives;
    }
}
