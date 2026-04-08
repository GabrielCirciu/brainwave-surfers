using UnityEngine;

public class Obstacle : MonoBehaviour
{
    private float moveSpeed = 10.0f;
    private float despawnZ = -10.0f;

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
            Destroy(gameObject);
        }
    }

    // Check if collided with the player
    void OnTriggerEnter(Collider objectCollider)
    {
        ScoreManager.Instance.EndGame();
    }
}
