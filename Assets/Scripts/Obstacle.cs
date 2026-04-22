using UnityEngine;

public class Obstacle : MonoBehaviour
{
    [SerializeField] private VehicleMover vehicleMover;
    [SerializeField] private LifeManager lifeManager;
    private float moveSpeed = 10.0f;
    private float despawnZ = -10.0f;

    void Start()
    {
        vehicleMover = FindFirstObjectByType<VehicleMover>();
        lifeManager = FindFirstObjectByType<LifeManager>();
    }

    public void SetSpeed(float speed)
    {
        moveSpeed = speed;
    }

    void Update()
    {
        // Move backwards along Z axis
        transform.Translate(Vector3.down * moveSpeed * Time.deltaTime);

        // Add score
        if (transform.position.z <= despawnZ)
        {
            ScoreManager.Instance.AddPoint();
            vehicleMover.MoveToOrigin(0.5f);
            Destroy(gameObject);
        }
    }

    // Check if collided with the player
    void OnTriggerEnter(Collider objectCollider)
    {
        Debug.Log("Collision!");
        lifeManager.LoseLife();
        Destroy(gameObject);
        vehicleMover.MoveToOrigin(0.5f);
    }
}
