using UnityEngine;

public class ObstacleSpawner : MonoBehaviour
{
    [Header("Spawner Settings")]
    public GameObject obstaclePrefab;
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

        Instantiate(obstaclePrefab, spawnPos, Quaternion.identity);
    }
}
