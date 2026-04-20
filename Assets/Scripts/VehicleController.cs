using UnityEngine;

public class VehicleController : MonoBehaviour
{
    [SerializeField] private VehicleMover vehicleMover;

    void Start()
    {

    }

    void Update()
    {
        // Using legacy input to be compatible with GTec
        if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A))
        {
            vehicleMover.MoveLeft(0.5f);
        }
        else if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D))
        {
            vehicleMover.MoveRight(0.5f);
        }

    }
}
