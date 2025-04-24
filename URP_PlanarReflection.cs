using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Experimental.Rendering;

[ExecuteInEditMode]
[RequireComponent(typeof(MeshRenderer))]
public class URP_PlanarReflection : MonoBehaviour
{
    [Header("Reflection Settings")]
    public LayerMask reflectionMask = -1;
    public bool reflectSkybox = false;
    public float clipPlaneOffset = 0.07f;
    public bool useHDR = true;

    [Header("Blur Settings")]
    public bool enableBlur = true;
    [Range(0.0f, 5.0f)] public float blurSize = 1.0f;
    [Range(0, 10)] public int blurIterations = 2;
    [Range(1.0f, 4.0f)] public float downsample = 1.0f;

    private const string ReflectionTexName = "_ReflectionTex";
    private Camera reflectionCamera;
    private RenderTexture reflectionRT;
    private RenderTexture blurredRT;
    private Material reflectionMaterial;
    private static bool isRendering;

    // Blur resources
    private Material blurMaterial;
    private Shader blurShader;

    void Start()
    {
        reflectionMaterial = GetComponent<MeshRenderer>().sharedMaterial;
        blurShader = Shader.Find("Hidden/KawaseBlur");
        blurMaterial = new Material(blurShader);
    }

    void OnEnable()
    {
        RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
    }

    void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
        SafeReleaseRT(ref reflectionRT);
        SafeReleaseRT(ref blurredRT);
        if (reflectionCamera) DestroyImmediate(reflectionCamera.gameObject);
    }

    private void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        if (isRendering || camera.cameraType != CameraType.Game)
            return;

        PrepareReflectionCamera(camera);
        RenderReflection(context, camera);
        ApplyPostProcessing(camera);
    }

    private void PrepareReflectionCamera(Camera mainCamera)
    {
        if (!reflectionCamera)
        {
            GameObject go = new GameObject($"Reflection Camera - {mainCamera.name}");
            go.hideFlags = HideFlags.HideAndDontSave;
            reflectionCamera = go.AddComponent<Camera>();
            reflectionCamera.depth = mainCamera.depth - 1;
        }

        reflectionCamera.CopyFrom(mainCamera);
        reflectionCamera.cullingMask = reflectionMask;
        reflectionCamera.clearFlags = reflectSkybox ? CameraClearFlags.Skybox : CameraClearFlags.SolidColor;
        reflectionCamera.depthTextureMode = DepthTextureMode.None;
        reflectionCamera.enabled = false;

        UpdateRenderTexture(mainCamera);
    }

    private void UpdateRenderTexture(Camera sourceCam)
    {
        int width = (int)(sourceCam.pixelWidth / downsample);
        int height = (int)(sourceCam.pixelHeight / downsample);
        RenderTextureFormat format = useHDR ? RenderTextureFormat.DefaultHDR : RenderTextureFormat.Default;

        if (!reflectionRT || reflectionRT.width != width || reflectionRT.height != height)
        {
            SafeReleaseRT(ref reflectionRT);
            reflectionRT = new RenderTexture(width, height, 24, format);
            reflectionRT.name = "PlanarReflectionRT";
        }

        reflectionCamera.targetTexture = reflectionRT;
    }

    private void RenderReflection(ScriptableRenderContext context, Camera mainCamera)
    {
        isRendering = true;
        
        // Calculate reflection matrix
        Vector3 normal = transform.up;
        Vector3 pos = transform.position;
        float d = -Vector3.Dot(normal, pos) - clipPlaneOffset;
        Vector4 plane = new Vector4(normal.x, normal.y, normal.z, d);
        Matrix4x4 reflectionMat = Matrix4x4.identity;
        reflectionMat = CalculateReflectionMatrix(reflectionMat, plane);

        // Setup reflection camera
        reflectionCamera.transform.position = reflectionMat.MultiplyPoint(mainCamera.transform.position);
        reflectionCamera.transform.eulerAngles = new Vector3(
            -mainCamera.transform.eulerAngles.x,
            mainCamera.transform.eulerAngles.y,
            mainCamera.transform.eulerAngles.z
        );

        // Update projection matrix
        reflectionCamera.worldToCameraMatrix = mainCamera.worldToCameraMatrix * reflectionMat;
        Vector4 clipPlane = CameraSpacePlane(reflectionCamera, pos, normal, 1.0f);
        reflectionCamera.projectionMatrix = mainCamera.CalculateObliqueMatrix(clipPlane);

        // Invert culling
        bool originalInvertCulling = GL.invertCulling;
        GL.invertCulling = true;

        // URP rendering
        UniversalRenderPipeline.RenderSingleCamera(context, reflectionCamera);
        
        

        GL.invertCulling = originalInvertCulling;
        isRendering = false;
    }

    private void ApplyPostProcessing(Camera mainCamera)
    {
        if (enableBlur)
        {
            if (!blurredRT || blurredRT.width != reflectionRT.width || blurredRT.height != reflectionRT.height)
            {
                SafeReleaseRT(ref blurredRT);
                blurredRT = new RenderTexture(reflectionRT);
            }

            RenderTexture tempRT = RenderTexture.GetTemporary(reflectionRT.descriptor);
            Graphics.Blit(reflectionRT, tempRT);

            for (int i = 0; i < blurIterations; i++)
            {
                float offset = i * 1.0f / downsample + blurSize;
                blurMaterial.SetFloat("_Offset", offset);
                
                Graphics.Blit(tempRT, blurredRT, blurMaterial);
                Graphics.Blit(blurredRT, tempRT);
            }

            RenderTexture.ReleaseTemporary(tempRT);
            reflectionMaterial.SetTexture(ReflectionTexName, blurredRT);
        }
        else
        {
            reflectionMaterial.SetTexture(ReflectionTexName, reflectionRT);
        }
    }

    private Matrix4x4 CalculateReflectionMatrix(Matrix4x4 matrix, Vector4 plane)
    {
        matrix.m00 = 1 - 2 * plane[0] * plane[0];
        matrix.m01 = -2 * plane[0] * plane[1];
        matrix.m02 = -2 * plane[0] * plane[2];
        matrix.m03 = -2 * plane[3] * plane[0];

        matrix.m10 = -2 * plane[1] * plane[0];
        matrix.m11 = 1 - 2 * plane[1] * plane[1];
        matrix.m12 = -2 * plane[1] * plane[2];
        matrix.m13 = -2 * plane[3] * plane[1];

        matrix.m20 = -2 * plane[2] * plane[0];
        matrix.m21 = -2 * plane[2] * plane[1];
        matrix.m22 = 1 - 2 * plane[2] * plane[2];
        matrix.m23 = -2 * plane[3] * plane[2];

        return matrix;
    }

    private Vector4 CameraSpacePlane(Camera cam, Vector3 pos, Vector3 normal, float sign)
    {
        Matrix4x4 m = cam.worldToCameraMatrix;
        Vector3 cpos = m.MultiplyPoint(pos + normal * clipPlaneOffset);
        Vector3 cnormal = m.MultiplyVector(normal).normalized * sign;
        return new Vector4(cnormal.x, cnormal.y, cnormal.z, -Vector3.Dot(cpos, cnormal));
    }

    private void SafeReleaseRT(ref RenderTexture rt)
    {
        if (rt)
        {
            rt.Release();
            DestroyImmediate(rt);
            rt = null;
        }
    }
}