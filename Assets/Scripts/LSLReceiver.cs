using System;
using UnityEngine;
using LSL4Unity.Utils;

public class LSLReceiver : AFloatInlet
{
    public float[] LastSample { get; private set; }

    protected override void OnStreamAvailable()
    {
        LastSample = new float[ChannelCount];
    }

    protected override void Process(float[] newSample, double timestamp)
    {
        // Copy the new sample to LastSample for external access
        if (LastSample == null || LastSample.Length != newSample.Length)
        {
            LastSample = new float[newSample.Length];
        }
        Array.Copy(newSample, LastSample, newSample.Length);
    }
}
