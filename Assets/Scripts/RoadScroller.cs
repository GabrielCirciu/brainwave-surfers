using UnityEngine;

public class RoadScroller : MonoBehaviour
{
    [Header("Scrolling Settings")]
    [Tooltip("How fast the texture scrolls.")]
    public float X_scrollSpeed = 0.5f;
    public float Y_scrollSpeed = 0.5f;
    [Tooltip("The name of the texture property in the shader (usually _MainTex or _BaseMap).")]
    public string textureName = "_BaseMap";

    private Material roadMaterial;
    private float X_offset;
    private float Y_offset;

    void Start()
    {
        // Cache the material
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            roadMaterial = renderer.material;
        }
        else
        {
            Debug.LogError("RoadScroller: No Renderer found on " + gameObject.name);
            enabled = false;
        }
    }

    void Update()
    {
        // Increment offset over time
        X_offset += Time.deltaTime * X_scrollSpeed;
        Y_offset += Time.deltaTime * Y_scrollSpeed;
        
        // Apply offset to the material
        // We only scroll on the Y axis of the texture (which usually corresponds to the Z road direction)
        roadMaterial.SetTextureOffset(textureName, new Vector2(X_offset, Y_offset));
    }
}
