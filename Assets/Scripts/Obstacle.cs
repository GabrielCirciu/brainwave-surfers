using UnityEngine;

public class Obstacle : MonoBehaviour
{
    [SerializeField] private VehicleMover vehicleMover;
    private float moveSpeed = 10.0f;
    private float despawnZ = -10.0f;

    void Start()
    {
        vehicleMover = FindFirstObjectByType<VehicleMover>();
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
        ScoreManager.Instance.EndGame();
    }
}
