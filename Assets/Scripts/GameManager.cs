using UnityEngine;

public class GameManager : MonoBehaviour
{
    [Header("Game Objects")]
    [SerializeField] private GameObject MainMenuCanvas;
    [SerializeField] private GameObject ScoreManager;
    [SerializeField] private GameObject ObstacleSpawnwer;
    [SerializeField] private GameObject ScoreCanvas;
    [SerializeField] private GameObject BCIVisualERP3D;
    [SerializeField] private GameObject BCIMotionERP3D;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void onStartPressed()
    {
        Debug.Log("Start Pressed");
        MainMenuCanvas.SetActive(false);
        ScoreManager.SetActive(true);
        ObstacleSpawnwer.SetActive(true);
        ScoreCanvas.SetActive(true);

        // Depending on what the dropdown is, enable the corresponding BCI
        if (config == "Visual (M)")
        {
            
        }
        else if (config == "Visual (F)")
        {
            BCIVisualERP3D.SetActive(true);
        }
        else if (config == "Motion")
        {
            Debug.Log("Motion Imagery selected, not implemented yet");
        }
    }

    private string config = "Visual (M)"; // Default value

    // Unity dropdown with 3 options, Visual (M), Visual (F), Motion
    // Collect which dropdown menu is selected. 
    // Unity's OnValueChanged event for dropdowns passes an integer index by default!
    public void onConfigDropdownChanged(int index)
    {
        if (index == 0)
        {
            config = "Visual (M)";
            Debug.Log("Visual (M) selected");
            BCIMotionERP3D.SetActive(true);
            BCIVisualERP3D.SetActive(false);
            // Deactivate MI
        }
        else if (index == 1)
        {
            config = "Visual (F)";
            Debug.Log("Visual (F) selected");
            BCIMotionERP3D.SetActive(false);
            BCIVisualERP3D.SetActive(true);
            // Deactivate MI
        }
        else if (index == 2)
        {
            config = "Motion";
            Debug.Log("Motion selected");
            BCIMotionERP3D.SetActive(false);
            BCIVisualERP3D.SetActive(false);
            // Activate MI
        }
    }
}
