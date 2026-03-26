using UnityEngine;

public class Obstacle : MonoBehaviour
{
    [HideInInspector]
    public float moveSpeed = 10.0f;
    public float despawnZ = -10.0f;

    private bool scoreAdded = false;

    public void SetSpeed(float speed)
    {
        moveSpeed = speed;
    }

    void Update()
    {
        // Move backwards along Z axis
        transform.Translate(Vector3.down * moveSpeed * Time.deltaTime);

        // Check if passed the player safely
        if (transform.position.z <= despawnZ && !scoreAdded)
        {
            scoreAdded = true;
            if (ScoreManager.Instance != null)
            {
                ScoreManager.Instance.AddPoint();
            }
            Destroy(gameObject);
        }
    }

    void OnTriggerEnter(Collider healthcare)
    {
        // Check if collided with the player
        if (healthcare.CompareTag("Player"))
        {
            Debug.Log("Collision! Game Over or Score Reset.");
            ScoreManager.Instance.ResetScore();
            Destroy(gameObject);
        }
    }
}
