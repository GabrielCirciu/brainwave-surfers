using System;
using UnityEngine;
using LSL;
using System.Collections;

public class LSLReceiver : MonoBehaviour
{
    public float[] LastSample { get; private set; }
    private StreamInlet inlet;

    void Start()
    {
        LastSample = new float[3];
        StartCoroutine(ResolveStreamRoutine());
    }

    IEnumerator ResolveStreamRoutine()
    {
        while (inlet == null)
        {
            StreamInfo[] results = LSL.LSL.resolve_stream("name", "BCIPredictor3", 1, 0.0);
            if (results.Length > 0)
            {
                inlet = new StreamInlet(results[0]);
                Debug.Log("Found and connected to BCIPredictor3 Stream!");
            }
            yield return new WaitForSeconds(1f);
        }
    }

    void Update()
    {
        if (inlet != null)
        {
            try
            {
                float[] sample = new float[3];
                // Non-blocking pull
                double ts = inlet.pull_sample(sample, 0.0);
                if (ts != 0.0)
                {
                    LastSample = sample;
                    // Flush the rest to get the absolute newest prediction
                    while (inlet.samples_available() > 0)
                    {
                        inlet.pull_sample(sample, 0.0);
                        LastSample = sample;
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("Lost connection to stream: " + e.Message);
                inlet = null;
                StartCoroutine(ResolveStreamRoutine());
            }
        }
    }
}
