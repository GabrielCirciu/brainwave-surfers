using UnityEngine;

public class VehicleController : MonoBehaviour
{
    [Header("Movement Settings")]
    [Tooltip("The X position for the left side of the road.")]
    public float leftX = -2.0f;
    [Tooltip("The X position for the right side of the road.")]
    public float rightX = 2.0f;
    [Tooltip("How fast the vehicle moves between sides.")]
    public float moveSpeed = 10.0f;

    private bool isOnRightSide = false;
    private float targetX;

    void Start()
    {
        // Initialize position
        targetX = leftX;
        Vector3 pos = transform.position;
        pos.x = leftX;
        transform.position = pos;
    }

    void Update()
    {
        // Legacy Input System: Horizontal axis or specific keys
        if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A))
        {
            isOnRightSide = false;
            targetX = leftX;
        }
        else if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D))
        {
            isOnRightSide = true;
            targetX = rightX;
        }

        // Smoothly move towards target X
        Vector3 currentPos = transform.position;
        float newX = Mathf.Lerp(currentPos.x, targetX, Time.deltaTime * moveSpeed);
        transform.position = new Vector3(newX, currentPos.y, currentPos.z);
    }
}
