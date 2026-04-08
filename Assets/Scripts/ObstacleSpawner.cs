using UnityEngine;

public class ObstacleSpawner : MonoBehaviour
{
    [SerializeField] private GameObject obstaclePrefab;

    private float spawnTimer;
    private float speedTimer;

    private float spawnInterval = 5.0f; // in seconds
    private float spawnIntervalIncreaseAmount = 1.0f;
    private float speedIncreaseInterval = 10f; // in seconds
    private float speedIncreaseAmount = 1f;

    private float obstacleSpeed = 10.0f;
    private float leftX = -7.0f;
    private float rightX = 7.0f;
    private float spawnZ = 200.0f;

    private bool isLeftLane = true;
    
    // Called on start, sets initial timers
    void Start()
    {
        spawnTimer = spawnInterval;
        speedTimer = speedIncreaseInterval;
    }

    // Called every frame, timers and speed increases
    void Update()
    {
        spawnTimer -= Time.deltaTime;
        speedTimer -= Time.deltaTime;

        if (spawnTimer <= 0f)
        {
            SpawnObstacle();
            spawnTimer = spawnInterval;
        }

        if (speedTimer <= 0f)
        {
            obstacleSpeed += speedIncreaseAmount;
            spawnInterval += spawnIntervalIncreaseAmount;
            speedTimer = speedIncreaseInterval;
        }   
    }

    // Spawn object on a opposite lane, rotate it, and set its speed
    void SpawnObstacle()
    {
        float spawnX = isLeftLane ? leftX : rightX;
        isLeftLane = !isLeftLane;
        Vector3 spawnPos = new Vector3(spawnX, 0.6f, spawnZ); 
        GameObject go = Instantiate(obstaclePrefab, spawnPos, Quaternion.Euler(90, 0, 0));
        Obstacle obstacle = go.GetComponent<Obstacle>();
        obstacle.SetSpeed(obstacleSpeed);
    }
}
