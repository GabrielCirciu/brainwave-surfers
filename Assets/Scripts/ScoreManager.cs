using UnityEngine;
using TMPro;

public class ScoreManager : MonoBehaviour
{
    public static ScoreManager Instance { get; private set; }
    public TextMeshProUGUI scoreText;
    public GameObject ObstacleSpawnwer;

    public int score = 0;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            scoreText.text = score.ToString();
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public void AddPoint()
    {
        score++;
        // Debug.Log("Score: " + score);
        scoreText.text = score.ToString();
    }

    public int GetScore()
    {
        return score;
    }

    public void ResetScore()
    {
        score = 0;
        scoreText.text = score.ToString();
        ObstacleSpawnwer.SetActive(false);
        // Find all objects with tag "Vehicle" and destroy them
        GameObject[] vehicles = GameObject.FindGameObjectsWithTag("Vehicle");
        foreach (GameObject vehicle in vehicles)
        {
            Destroy(vehicle);
        }
    }


}
