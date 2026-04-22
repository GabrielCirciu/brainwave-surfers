using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using TMPro;

public class CalibrationManager : MonoBehaviour
{
    [Header("LSL")]
    public LSLMarkerStream markerStream;
    public TextMeshProUGUI promptText;

    [Header("Vehicle")]
    [SerializeField] private VehicleMover vehicleMover;
    [SerializeField] private GameObject MainMenuPanel;
    [SerializeField] private GameObject CalibrationButton;
    [SerializeField] private GameObject CalibrationText;
    [SerializeField] private GameObject CalibrationPlayButton;
    [SerializeField] private GameObject ScoreCanvas;

    private int trialsPerClass = 5;
    private float relaxDuration = 3.0f;
    
    private bool isCalibrating = false;
    private bool isPaused = false;
    private bool discardCurrentTrial = false;
    private string fun_fact = "";

    private void Update()
    {
        if (isCalibrating && Input.GetKeyDown(KeyCode.Space))
        {
            isPaused = !isPaused;
            if (isPaused)
            {
                discardCurrentTrial = true;
            }
        }
    }

    public void StartCalibration()
    {
        isCalibrating = true;
        isPaused = false;
        discardCurrentTrial = false;
        Random.InitState(42); // Sets random seed. In the future we will have different seeds per trial
        StartCoroutine(CalibrationRoutine());
        CalibrationButton.SetActive(false);
        CalibrationPlayButton.SetActive(false);
    }

    private string GetFunFact()
    {
        string[] funFacts = new string[]
        {
            "The average person walks the equivalent of three times around the world in their lifetime.",
            "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
            "The world’s quietest room is at Microsoft’s headquarters in Redmond, Washington. It’s so quiet that the background noise is measured in negative decibels, and people can actually hear their own heartbeat and lungs working.",
            "Octopuses have three hearts. Two pump blood through the gills, while the third circulates blood to the rest of the body.",
            "A group of flamingos is called a flamboyance.",
            "The Eiffel Tower can be 15 cm taller during the summer due to the thermal expansion of the iron.",
            "The smell of freshly cut grass is actually a distress signal released by the plants.",
            "The human nose can detect over 1 trillion different scents.",
            "Bananas are berries, but strawberries are not.",
            "The Great Barrier Reef is the largest living structure on Earth, visible from space."
        };
        return funFacts[Random.Range(0, funFacts.Length)];
    }

    private IEnumerator CalibrationRoutine()
    {
        promptText.text = "Greetings commander!\n\nYou are about to embark on a thrilling journey where your mind controls your spaceship.\n\nBefore we begin, we need to calibrate your neural signals.";
        
        float startTimer = 0;
        while (startTimer < 10f)
        {
            if (isPaused)
            {
                promptText.text = "Paused. Press Space to resume.";
                while (isPaused) yield return null;
                promptText.text = "Greetings commander!\n\nYou are about to embark on a thrilling journey where your mind controls your spaceship.\n\nBefore we begin, we need to calibrate your neural signals.";
            }
            startTimer += Time.deltaTime;
            yield return null;
        }

        promptText.text = "We will calibrate your ship over several rounds.\n\nEach round you will see an arrow point left or right, and for the duration of it, you must imagine what it feels like to pull the ship towards that direction.";
        
        startTimer = 0;
        while (startTimer < 15f)
        {
            if (isPaused)
            {
                promptText.text = "Paused. Press Space to resume.";
                while (isPaused) yield return null;
                promptText.text = "We will calibrate your ship over several rounds.\n\nEach round you will see an arrow point left or right, and for the duration of it, you must imagine what it feels like to pull the ship towards that direction.";
            }
            startTimer += Time.deltaTime;
            yield return null;
        }

        promptText.text = "Please sit still for the duration of the calibration.\n\nIt will take aproximately 10 minutes, with short breaks to relax.";
        
        startTimer = 0;
        while (startTimer < 10f)
        {
            if (isPaused)
            {
                promptText.text = "Paused. Press Space to resume.";
                while (isPaused) yield return null;
                promptText.text = "Please sit still for the duration of the calibration.\n\nIt will take aproximately 10 minutes, with short breaks to relax.";
            }
            startTimer += Time.deltaTime;
            yield return null;
        }

        promptText.text = "If you wish to pause the calibration, press the Spacebar. You can press it again to unpause.\n\nWe will begin now. Good luck!";
        
        startTimer = 0;
        while (startTimer < 10f)
        {
            if (isPaused)
            {
                promptText.text = "Paused. Press Space to resume.";
                while (isPaused) yield return null;
                promptText.text = "If you wish to pause the calibration, press the Spacebar. You can press it again to unpause.\n\nWe will begin now. Good luck!";
            }
            startTimer += Time.deltaTime;
            yield return null;
        }
        
        discardCurrentTrial = false;

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
            if (i > 0 && i % 10 == 0)
            {
                fun_fact = GetFunFact();
                float breakTimer = 0;
                while (breakTimer < 30f)
                {
                    if (isPaused)
                    {
                        promptText.text = "Paused. Press Space to resume.";
                        while (isPaused) yield return null;
                    }
                    breakTimer += Time.deltaTime;
                    promptText.text = (30 - breakTimer).ToString("F0") + " second break.\nTake a deep breath and relax.\nYou are doing great!\n\nFun fact:\n" + fun_fact;
                    yield return null;
                }

                // 5 second timer to let the user get back into a focused position
                float refocusTimer = 0;
                while (refocusTimer < 5f)
                {
                    if (isPaused)
                    {
                        promptText.text = "Paused. Press Space to resume.";
                        while (isPaused) yield return null;
                    }
                    refocusTimer += Time.deltaTime;
                    promptText.text = "Starting in " + (5 - refocusTimer).ToString("F0") + " seconds.\nPlease get back into a focused position.";
                    yield return null;
                }
            }

            discardCurrentTrial = false;

            promptText.text = "Relax";
            relaxDuration = Random.Range(4, 9) * 0.5f; // Picks 2.0, 2.5, 3.0, 3.5, or 4.0
            yield return StartCoroutine(Wait(relaxDuration));
            if (discardCurrentTrial) { i--; yield return StartCoroutine(HandlePause()); continue; }

            promptText.text = "Ready...";
            yield return StartCoroutine(Wait(1.5f)); // Get ready timer before showing cue
            if (discardCurrentTrial) { i--; yield return StartCoroutine(HandlePause()); continue; }

            string dir = "";
            if (trialQueue[i] == 0) dir = "< -        ";
            else if (trialQueue[i] == 1) dir = "        - >";
            promptText.text = dir;
            yield return StartCoroutine(Wait(1f)); // Cue timer shown for 1 second before recording
            if (discardCurrentTrial) { i--; yield return StartCoroutine(HandlePause()); continue; }
            
            if (dir == "< -        ")
            {
                markerStream.WriteMarker("LEFT_START");
                vehicleMover.MoveLeft(4.0f);
            }
            else if (dir == "        - >")
            {
                markerStream.WriteMarker("RIGHT_START");
                vehicleMover.MoveRight(4.0f);
            }

            yield return StartCoroutine(Wait(4.0f)); // Task timer
            if (discardCurrentTrial) { i--; yield return StartCoroutine(HandlePause()); continue; }

            if (dir == "< -        ")
            {
                markerStream.WriteMarker("LEFT_END");
                vehicleMover.ReturnToOrigin();
            }
            else if (dir == "        - >")
            {
                markerStream.WriteMarker("RIGHT_END");
                vehicleMover.ReturnToOrigin();
            }
        }

        isCalibrating = false;
        promptText.text = "Done!";
        markerStream.WriteMarker("CALIBRATION_END");
        StartGame();
    }

    private IEnumerator Wait(float seconds)
    {
        float timer = 0;
        while (timer < seconds)
        {
            if (discardCurrentTrial)
            {
                yield break;
            }
            timer += Time.deltaTime;
            yield return null;
        }
    }

    private IEnumerator HandlePause()
    {
        markerStream.WriteMarker("DISCARD");
        vehicleMover.ReturnToOrigin();
        promptText.text = "Paused. Press Space to resume.";
        
        while (isPaused)
        {
            yield return null;
        }
        
        discardCurrentTrial = false;
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
