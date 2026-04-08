using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using TMPro;

public class CalibrationManager : MonoBehaviour
{
    [Header("LSL")]
    public LSLMarkerStream markerStream;
    public TextMeshProUGUI promptText;
    
    [Header("Settings")]
    public int trialsPerClass = 15;
    public float relaxDuration = 2.0f;

    [Header("Vehicle")]
    [SerializeField] private VehicleMover vehicleMover;
    [SerializeField] private GameObject MainMenuPanel;
    [SerializeField] private GameObject CalibrationButton;
    [SerializeField] private GameObject CalibrationText;
    [SerializeField] private GameObject CalibrationPlayButton;
    [SerializeField] private GameObject ScoreCanvas;

    public void StartCalibration()
    {
        StartCoroutine(CalibrationRoutine());
        CalibrationButton.SetActive(false);
        CalibrationPlayButton.SetActive(false);
    }

    private IEnumerator CalibrationRoutine()
    {
        // Initialize LSL Python script here within 5 seconds
        promptText.text = "Starting soon...";
        yield return new WaitForSeconds(5f);

        int totalTrials = trialsPerClass * 2;
        int[] trialQueue = new int[totalTrials];
        for (int i = 0; i < trialsPerClass; i++)
        {
            trialQueue[i] = 0; // Left
            trialQueue[i + trialsPerClass] = 1; // Right
        }

        // Shuffle trial order
        for (int i = 0; i < totalTrials; i++)
        {
            int temp = trialQueue[i];
            int r = Random.Range(i, totalTrials);
            trialQueue[i] = trialQueue[r];
            trialQueue[r] = temp;
        }

        for (int i = 0; i < totalTrials; i++)
        {
            promptText.text = "Relax";
            relaxDuration = Random.Range(2.0f, 4.0f); // Random range of relax timer
            yield return new WaitForSeconds(relaxDuration);
            promptText.text = "Ready...";
            yield return new WaitForSeconds(1.5f); // Get ready timer before showing cue
            string dir = trialQueue[i] == 0 ? "<-" : "->";
            promptText.text = dir;
            yield return new WaitForSeconds(1f); // Show cue timer before recording
            vehicleMover.MoveInXSeconds(dir, 4.0f);
            if (dir == "<-")
            {
                markerStream.WriteMarker("LEFT_START");
            }
            else
            {
                markerStream.WriteMarker("RIGHT_START");
            }
            yield return new WaitForSeconds(4.0f); // Task timer
            if (dir == "<-")
            {
                markerStream.WriteMarker("LEFT_END");
            }
            else
            {
                markerStream.WriteMarker("RIGHT_END");
            }
            vehicleMover.ReturnToOrigin();
        }

        promptText.text = "Done!";
        markerStream.WriteMarker("CALIBRATION_END");
        StartGame();
    }

    public void StartGame()
    {
        MainMenuPanel.SetActive(true);
        CalibrationButton.SetActive(false);
        CalibrationText.SetActive(false);
        CalibrationPlayButton.SetActive(false);
        ScoreCanvas.SetActive(true);
    }
}
