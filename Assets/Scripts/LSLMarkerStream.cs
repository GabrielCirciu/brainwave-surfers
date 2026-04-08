using UnityEngine;
using LSL;

public class LSLMarkerStream : MonoBehaviour
{
    private StreamOutlet outlet;
    private string[] sample = new string[1];
    private bool isInitialized = false;

    void Start()
    {
        try
        {
            StreamInfo streamInfo = new StreamInfo("UnityMarkers", "Markers", 1, LSL.LSL.IRREGULAR_RATE, channel_format_t.cf_string, "unity_markers");
            outlet = new StreamOutlet(streamInfo);
            isInitialized = true;
            Debug.Log("LSL Marker Stream Created: UnityMarkers");
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to create LSL Marker Stream. Make sure LSL libs are loaded. Error: " + e.Message);
        }
    }

    public void WriteMarker(string marker)
    {
        if (isInitialized && outlet != null)
        {
            sample[0] = marker;
            outlet.push_sample(sample);
            // Debug.Log("Sent Marker: " + marker);
        }
    }
}
