using UnityEngine;

public class VehicleController : MonoBehaviour
{
    private float leftX = -7.0f;
    private float rightX = 7.0f;
    private float moveSpeed = 10.0f;

    [Header("LSL Settings")]
    public LSLReceiver lslReceiver;
    public int leftChannelIndex = 0;   // Example indexes
    public int rightChannelIndex = 1;
    public float threshold = 0.70f;     // Threshold for MI probability (0.0 to 1.0)
    
    private float lastMoveTime = 0f;
    private float moveCooldown = 1.0f; // Prevent rapid swapping lanes

    private float targetX;
    private float targetY = 0f;

    void Start()
    {
        targetX = leftX;
    }

    void Update()
    {
        // Using legacy input to be compatible with GTec
        if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A))
        {
            targetX = leftX;
        }
        else if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D))
        {
            targetX = rightX;
        }

        // LSL Input Logic
        if (lslReceiver != null && lslReceiver.LastSample != null && lslReceiver.LastSample.Length >= 2)
        {
            if (Time.time - lastMoveTime > moveCooldown)
            {
                float leftProb = lslReceiver.LastSample[leftChannelIndex];
                float rightProb = lslReceiver.LastSample[rightChannelIndex];

                if (leftProb > threshold && leftProb > rightProb)
                {
                    targetX = leftX;
                    lastMoveTime = Time.time;
                    Debug.Log($"BCI MOVE LEFT! Prob: {(leftProb*100):0.0}%");
                }
                else if (rightProb > threshold && rightProb > leftProb)
                {
                    targetX = rightX;
                    lastMoveTime = Time.time;
                    Debug.Log($"BCI MOVE RIGHT! Prob: {(rightProb*100):0.0}%");
                }
            }
        }

        // Smoothly move towards target X coordinate
        Vector3 currentPos = transform.position;
        float newX = Mathf.Lerp(currentPos.x, targetX, Time.deltaTime * moveSpeed);
        transform.position = new Vector3(newX, targetY, currentPos.z);
    }
}
