using UnityEngine;

public class ObstacleSpawner : MonoBehaviour
{
    [Header("Spawner Settings")]
    public GameObject obstaclePrefab;
    public float obstacleSpeed = 10.0f;
    public float spawnInterval = 2.0f;
    public float spawnZ = 50.0f;
    
    [Header("Lane Settings")]
    public float leftX = -2.0f;
    public float rightX = 2.0f;

    private float timer;

    void Start()
    {
        timer = spawnInterval;
    }

    void Update()
    {
        timer -= Time.deltaTime;

        if (timer <= 0f)
        {
            SpawnObstacle();
            timer = spawnInterval;
        }
    }

    void SpawnObstacle()
    {
        if (obstaclePrefab == null) return;

        // Randomly choose a lane (left or right)
        float spawnX = Random.value > 0.5f ? rightX : leftX;
        Vector3 spawnPos = new Vector3(spawnX, 0.6f, spawnZ); // Adjust Y based on plane/cube size
        // rotate object on X axis by 90 degrees
        GameObject go = Instantiate(obstaclePrefab, spawnPos, Quaternion.Euler(90, 0, 0));
        Obstacle obstacle = go.GetComponent<Obstacle>();
        obstacle.SetSpeed(obstacleSpeed);
    }
}
