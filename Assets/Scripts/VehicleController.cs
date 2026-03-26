using UnityEngine;

public class VehicleController : MonoBehaviour
{
    [Header("Movement Settings")]
    [Tooltip("The X position for the left side of the road.")]
    public float leftX = -5.0f;
    [Tooltip("The X position for the right side of the road.")]
    public float rightX = 5.0f;
    [Tooltip("How fast the vehicle moves between sides.")]
    public float moveSpeed = 10.0f;

    [Header("LSL Settings")]
    public LSLReceiver lslReceiver;
    public int leftChannelIndex = 0;   // Example indexes
    public int rightChannelIndex = 1;
    public float threshold = 0.5f;     // Threshold for activation

    private float targetX;

    void Start()
    {
        // Initialize position
        targetX = leftX;
    }

    void Update()
    {
        // Legacy Input System: Horizontal axis or specific keys
        if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A))
        {
            targetX = leftX;
        }
        else if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D))
        {
            targetX = rightX;
        }

        // LSL Input Logic
        if (lslReceiver != null && lslReceiver.LastSample != null)
        {
            Debug.Log("Left: " + lslReceiver.LastSample[leftChannelIndex] + " Right: " + lslReceiver.LastSample[rightChannelIndex]);

            if (lslReceiver.LastSample[leftChannelIndex] > threshold)
            {
                targetX = leftX;
            }
            else if (lslReceiver.LastSample[rightChannelIndex] > threshold)
            {
                targetX = rightX;
            }
        }

        // Smoothly move towards target X
        Vector3 currentPos = transform.position;
        float newX = Mathf.Lerp(currentPos.x, targetX, Time.deltaTime * moveSpeed);
        transform.position = new Vector3(newX, currentPos.y, currentPos.z);
    }
}
