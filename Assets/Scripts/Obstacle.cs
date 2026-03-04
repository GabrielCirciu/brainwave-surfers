using UnityEngine;

public class Obstacle : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 10.0f;
    public float despawnZ = -10.0f;

    private bool scoreAdded = false;

    void Update()
    {
        // Move backwards along Z axis
        transform.Translate(Vector3.back * moveSpeed * Time.deltaTime);

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
            ScoreManager.Instance.score = 0;
            Destroy(gameObject);
        }
    }
}
