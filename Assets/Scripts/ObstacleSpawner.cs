using UnityEngine;

public class ObstacleSpawner : MonoBehaviour
{
    [SerializeField] private GameObject obstaclePrefab;
    [SerializeField] private LSLMarkerStream markerStream;

    private float spawnTimer;
    private float sleepTimer;

    private float[] spawnIntervals = {1.0f, 8.0f, 7.5f, 7.0f, 6.5f, 6.0f, 5.5f, 5.0f};
    private int spawnIntervalIndex = 0;
    private float[] sleepIntervals = {1.5f, 8.0f, 7.5f, 7.0f, 6.5f, 6.0f, 5.5f, 5.0f};
    private int sleepIntervalIndex = 0;

    private float obstacleSpeed = 40.0f;
    private float leftX = -7.0f;
    private float rightX = 7.0f;
    private float spawnZ = 200.0f;

    private bool isLeftLane = true;
    
    // Called on start, sets initial timers
    void Start()
    {
        spawnTimer = spawnIntervals[spawnIntervalIndex];
        sleepTimer = sleepIntervals[sleepIntervalIndex];
    }

    // Called every frame, timers and speed increases
    void Update()
    {
        spawnTimer -= Time.deltaTime;
        if (spawnTimer <= 0f)
        {
            SpawnObstacle();
            // Progress to the next interval step, capping at the final step (5.0f)
            if (spawnIntervalIndex < spawnIntervals.Length - 1)
            {
                spawnIntervalIndex++;
            }
            spawnTimer = spawnIntervals[spawnIntervalIndex];
        }

        sleepTimer -= Time.deltaTime;
        if (sleepTimer <= 0f)
        {
            if (markerStream != null)
            {
                markerStream.WriteMarker("PREDICT_START");
                Debug.Log("PREDICT_START marker sent!");
            }
            else
            {
                Debug.Log("No marker stream found!");
            }
            if (sleepIntervalIndex < sleepIntervals.Length - 1)
            {
                sleepIntervalIndex++;
            }
            sleepTimer = sleepIntervals[sleepIntervalIndex];
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
