using UnityEngine;
using System.Collections;

public class VehicleMover : MonoBehaviour
{
    private Coroutine currentMoveCoroutine;

    public void MoveInXSeconds(string dir, float seconds)
    {
        if (currentMoveCoroutine != null) StopCoroutine(currentMoveCoroutine);
        Vector3 targetPos = new Vector3(transform.position.x + (dir == "<-" ? -7f : 7f), transform.position.y, transform.position.z);
        currentMoveCoroutine = StartCoroutine(MoveToPositionCoroutine(targetPos, seconds));
    }

    public void MoveLeft()
    {
        if (currentMoveCoroutine != null) StopCoroutine(currentMoveCoroutine);
        currentMoveCoroutine = StartCoroutine(MoveToPositionCoroutine(new Vector3(-7f, transform.position.y, transform.position.z), 4.0f));
    }

    public void MoveRight()
    {
        if (currentMoveCoroutine != null) StopCoroutine(currentMoveCoroutine);
        currentMoveCoroutine = StartCoroutine(MoveToPositionCoroutine(new Vector3(7f, transform.position.y, transform.position.z), 4.0f));
    }

    public void ReturnToOrigin()
    {
        if (currentMoveCoroutine != null) StopCoroutine(currentMoveCoroutine);
        currentMoveCoroutine = StartCoroutine(ReturnToOriginSequence());
    }

    private IEnumerator ReturnToOriginSequence()
    {
        Vector3 baseScale = Vector3.one;
        if (transform.localScale.magnitude > 0.1f) baseScale = transform.localScale;

        // 1. Scale down quickly over 0.15 seconds
        float duration = 0.2f;
        float timeElapsed = 0f;
        Vector3 startScale = transform.localScale;
        
        while (timeElapsed < duration)
        {
            transform.localScale = Vector3.Lerp(startScale, Vector3.one * 0.01f, timeElapsed / duration);
            timeElapsed += Time.deltaTime;
            yield return null;
        }

        // 2. Set to origin instantly while microscopic
        transform.position = new Vector3(0f, transform.position.y, transform.position.z);

        // 3. Scale back up slowly over 0.5 seconds
        duration = 0.3f;
        timeElapsed = 0f;
        
        while (timeElapsed < duration)
        {
            transform.localScale = Vector3.Lerp(Vector3.one * 0.01f, baseScale, timeElapsed / duration);
            timeElapsed += Time.deltaTime;
            yield return null;
        }

        transform.localScale = baseScale;
    }

    private IEnumerator MoveToPositionCoroutine(Vector3 targetPosition, float duration)
    {
        Vector3 startPosition = transform.position;
        float timeElapsed = 0f;

        while (timeElapsed < duration)
        {
            float t = timeElapsed / duration;
            // Make movement ramp up and slow down
            t = t * t * (3f - 2f * t);
            transform.position = Vector3.Lerp(startPosition, targetPosition, t);
            timeElapsed += Time.deltaTime;
            yield return null; // Wait for the next frame
        }

        transform.position = targetPosition; // Ensure it reaches the exact target position at the end
    }
}
