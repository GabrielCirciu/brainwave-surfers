using UnityEngine;

public class GameManager : MonoBehaviour
{
    [Header("Game Objects")]
    [SerializeField] private GameObject MainMenuCanvas;
    [SerializeField] private GameObject ScoreManager;
    [SerializeField] private GameObject ObstacleSpawnwer;
    [SerializeField] private GameObject BCIVisualERP3D;
    [SerializeField] private GameObject BCIMotionERP3D;
    [SerializeField] private GameObject LSLReceiver;
    [SerializeField] private GameObject PlayerVehicle;
    [SerializeField] private GameObject EnemyShip;
    private VehicleController vehicleController;

    void Awake()
    {
        vehicleController = PlayerVehicle.GetComponent<VehicleController>();
    }

    public void onStartPressed()
    {
        MainMenuCanvas.SetActive(false);
        ObstacleSpawnwer.SetActive(true);
        EnemyShip.SetActive(true);
        vehicleController.enabled = true;
        ScoreManager.GetComponent<ScoreManager>().StartGame();
    }

    public void onEndGame()
    {
        ObstacleSpawnwer.SetActive(false);
        GameObject[] obstacles = GameObject.FindGameObjectsWithTag("Obstacle");
        foreach (GameObject obstacle in obstacles)
        {
            Destroy(obstacle);
        }
        MainMenuCanvas.SetActive(true);
        vehicleController.enabled = false;
        PlayerVehicle.transform.position = new Vector3(0, 0, 0);
    }

    // Unity dropdown with 4 options, Nothing, Flash, Motion, Motor Imagery
    public void onConfigDropdownChanged(int index)
    {
        if (index == 0)
        {
            // Nothing, default state
        }
        else if (index == 1)
        {
            BCIMotionERP3D.SetActive(true);
            BCIVisualERP3D.SetActive(false);
            LSLReceiver.SetActive(false);
        }
        else if (index == 2)
        {
            BCIMotionERP3D.SetActive(false);
            BCIVisualERP3D.SetActive(true);
            LSLReceiver.SetActive(false);
        }
        else if (index == 3)
        {
            BCIMotionERP3D.SetActive(false);
            BCIVisualERP3D.SetActive(false);
            LSLReceiver.SetActive(true);
        }
    }
}
