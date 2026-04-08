using UnityEngine;

public class RoadScroller : MonoBehaviour
{
    private Material roadMaterial;
    private float X_offset;
    private float Y_offset;

    void Start()
    {
        // Cache the material
        roadMaterial = GetComponent<Renderer>().material;
    }

    void Update()
    {
        // Increment offset over time
        X_offset += Time.deltaTime * 0;
        Y_offset += Time.deltaTime * 0.1f;
        
        // Apply offset to the material
        // We only scroll on the Y axis of the texture (which usually corresponds to the Z road direction)
        roadMaterial.SetTextureOffset("_BaseMap", new Vector2(X_offset, Y_offset));
    }
}
