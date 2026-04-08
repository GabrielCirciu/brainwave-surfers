using UnityEngine;
using TMPro;

public class ScoreManager : MonoBehaviour
{
    public static ScoreManager Instance { get; private set; }
    [SerializeField] private TextMeshProUGUI scoreText;
    [SerializeField] private GameObject GameManager;
    private GameManager gameManager;
    private int score = 0;

    void Awake()
    {
        Instance = this;
        gameManager = GameManager.GetComponent<GameManager>();
    }

    public void StartGame()
    {
        score = 0;
        scoreText.text = score.ToString();
    }

    public void AddPoint()
    {
        score++;
        scoreText.text = score.ToString();
    }

    public void EndGame()
    {
        gameManager.onEndGame();
    }


}
