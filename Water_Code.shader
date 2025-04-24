Shader "Unlit/Water_Code_URP"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _MainTex_ST ("Texture ST", Vector) = (1,1,0,0)
        _DeepColor ("Deep Color", Color) = (0,0,0,1)
        _ShallowColor ("Shallow Color", Color) = (0,0,0,1)
        _DeepRange ("Deep Range", Range(-5, 5)) = 1
        _FresnelColor ("Fresnel Color", Color) = (0,0,0,1)
        _FresnelPower ("Fresnel Power", Range(0, 50)) = 1
        _FresnelIntensity ("Fresnel Intensity", Range(0, 10)) = 1.0

        _NormalMap ("Normal Map", 2D) = "bump" {}
        _BumpScale ("Bump Scale", Float) = 1.0
        _NormalSpeed ("Normal Speed", Vector) = (1, 1, 0, 0)

        _ReflectionTex ("Reflection", 2D) = "white" {}
        _ReflectDistort ("Reflection Distort", Range(-5, 5)) = 1.0
        _ReflectIntensity ("Reflection Intensity", Range(0, 10)) = 1.0
        _ReflectPower ("Reflection Power", Range(0, 50)) = 5.0

        _UnderWaterDistort ("Under Water Distort", Range(-5, 5)) = 1.0

        _CausticsTex ("Caustics", 2D) = "white" {} 
        _CausticsScale("Caustics Scale",Float) = 8
        _CausticsSpeed ("Caustics Speed", Vector) = (-8, 0, 0, 0)
        _CausticsIntensity ("Caustics Intensity", Range(0, 10)) = 1.0
        _CausticsRange ("Caustics Range", Range(0, 50)) = 1.0

        _ShoreRange ("shore Range", Range(0, 50)) = 1
        _ShoreColor ("shore Color", Color) = (0,0,0,1)
        _ShoreEdgeWidth ("shore Edge Width", Range(0, 1)) = 1
        _ShoreIntensity ("shore Intensity", Range(0, 10)) = 1.0

        _FoamRange("Foam Range", Range(0, 50)) = 1
        _FoamSpeed("Foam Speed", Vector) = (1, 1, 0, 0)
        _FoamFrequency("Foam Frequency", Range(0, 50)) = 10
        _FoamBlend("Foam Blend", Range(0, 1)) = 1.0
        _NoiseTexture ("Noise Texture", 2D) = "white" {}
        _NoiseSize ("Noise Size", Vector) = (10, 10, 0, 0)
        _FoamDissolve ("Foam Dissolve", Range(0, 10)) = 1
        _FoamWidth ("Foam Width", Float) = 1.0
        _FoamColor ("Foam Color", Color) = (0,0,0,1)

        _WaveA ("Wave A (dir, steepness, wavelength)", Vector) = (1,0,0.5,10)
        _WaveB ("Wave B (dir, steepness, wavelength)", Vector) = (1,0,0.5,10)
        _WaveC ("Wave C (dir, steepness, wavelength)", Vector) = (1,0,0.5,10)
        _HighWaveColor ("High Wave Color", Color) = (0,0,0,1)
        _LowWaveColor ("Low Wave Color", Color) = (0,0,0,1)
        
        

        _GradientRange ("渐变范围", Range(0,50)) = 0.5
        _GradientOffset ("渐变偏移", Range(-50,50)) = 0
        _AlphaClip ("Alpha Clip", Range(0, 1)) = 1
    }

    SubShader
    {
        Tags 
        { 
            "RenderType" = "Transparent"
            "Queue" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
        }
        LOD 100

        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Back



        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/SurfaceInput.hlsl"

            float3 ReorientNormal(float3 base, float3 detail) {
                float3 t = base + float3(0.0, 0.0, 1.0);
                float3 u = detail * float3(-1.0, -1.0, 1.0);
                return normalize((t / t.z) * dot(t, u) - u);
            }


            struct Attributes
            {
                float4 positionOS : POSITION;
                float4 tangentOS : TANGENT;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD0;
                float2 texcoord : TEXCOORD1;

            };

            struct Varyings
            {
                float4 positionCS  : SV_POSITION;
                float2 uv : TEXCOORD0;
                float4 positionWSAndFogFactor : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float4 tangentWS : TEXCOORD3;
                float3 viewDirWS : TEXCOORD4;
                float4 screenPos : TEXCOORD5; // 新增语义索引 
                float3 bitangentWS : TEXCOORD6;
                float3 positionWS : TEXCOORD7;  
                float waveHeight : TEXCOORD8; // 新增：海浪高度
            };

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);
            TEXTURE2D(_NormalMap);
            SAMPLER(sampler_NormalMap);
            TEXTURE2D(_CameraOpaqueTexture);
            SAMPLER(sampler_CameraOpaqueTexture);
            sampler2D _ReflectionTex;
            sampler2D _CausticsTex;
            sampler2D _NoiseTexture;
             // 在 CBUFFER 外添加声明
            
            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                float4 _DeepColor;
                float4 _ShallowColor;
                float _DeepRange;
                float4 _FresnelColor;
                float _FresnelPower;
                float _AlphaClip;
                float _GradientRange;
                float _GradientOffset;
                float _BumpScale;
                float2 _NormalSpeed;
                float _ReflectDistort;
                float _UnderWaterDistort;
                float _FresnelIntensity;
                float _ReflectIntensity;
                float _ReflectPower;
                float _CausticsScale;
                float2 _CausticsSpeed;
                float _CausticsIntensity;
                float _CausticsRange;
                float _ShoreRange;
                float4 _ShoreColor;
                float _ShoreEdgeWidth;
                float _ShoreIntensity;
                float _FoamRange;
                float2 _FoamSpeed;
                float _FoamFrequency;
                float _FoamBlend;
                float2 _NoiseSize;
                float _FoamDissolve;
                float _FoamWidth;
                float4 _FoamColor;
                float4 _WaveA;
                float4 _WaveB;
                float4 _WaveC;

                float4 _HighWaveColor;
                float4 _LowWaveColor;

            CBUFFER_END

            float3 GerstnerWave(
            float4 wave, float3 p, inout float3 tangent, inout float3 binormal
        ) {
            // 从 wave 参数中提取波浪属性
            float steepness = wave.z;       // 波浪陡峭度
            float wavelength = wave.w;      // 波长
            float k = 2.0 * PI / wavelength; // 波数
            float c = sqrt(9.8 / k);        // 波速
            float2 d = normalize(wave.xy); // 波浪方向
            float f = k * (dot(d, p.xz) - c * _Time.y); // 波动函数
            float a = steepness / k;        // 振幅

            // 计算切线和副切线
            tangent += float3(
                -d.x * d.x * (steepness * sin(f)),
                d.x * (steepness * cos(f)),
                -d.x * d.y * (steepness * sin(f))
            );
            binormal += float3(
                -d.x * d.y * (steepness * sin(f)),
                d.y * (steepness * cos(f)),
                -d.y * d.y * (steepness * sin(f))
            );

            // 返回波浪的偏移
            return float3(
                d.x * (a * cos(f)),
                a * sin(f),
                d.y * (a * cos(f))
            );
        }

            Varyings vert (Attributes IN)
            {
                Varyings OUT;

                float3 gridPoint = IN.positionOS.xyz; // 获取顶点的初始位置
                
			    float3 tangent = float3(1, 0, 0);
			    float3 binormal = float3(0, 0, 1);// 初始化切线和副切线
                float3 accumulatedNormal = float3(0, 0, 0);

			    float3 p = gridPoint;
			    p += GerstnerWave(_WaveA, gridPoint, tangent, binormal);// 计算波浪偏移
                accumulatedNormal += normalize(cross(binormal, tangent)); // 累加法线
                p += GerstnerWave(_WaveB, gridPoint, tangent, binormal);//WaveB
                accumulatedNormal += normalize(cross(binormal, tangent)); // 累加法线
                p += GerstnerWave(_WaveC, gridPoint, tangent, binormal);//WaveC
                accumulatedNormal += normalize(cross(binormal, tangent)); // 累加法线


			    float3 normal = normalize(accumulatedNormal);

                // 计算海浪高度（对象空间到世界空间的 Y 轴差值）
                float waveHeight = p.y - gridPoint.y;

                OUT.waveHeight = waveHeight;

			    // 更新顶点位置和法线
                OUT.positionCS = TransformObjectToHClip(p);
                OUT.normalWS = normalize(TransformObjectToWorldNormal(normal));
                

                //OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.positionWS = TransformObjectToWorld(p);

                // 传递其他数据
                //OUT.bitangentWS = cross(IN.tangentOS.xyz, IN.normalOS);
                OUT.uv = TRANSFORM_TEX(IN.uv, _MainTex);

                OUT.positionWSAndFogFactor =  float4(TransformObjectToWorld(p.xyz), ComputeFogFactor(OUT.positionCS.z));
                
                //OUT.normalWS = normalize(TransformObjectToWorldNormal(IN.normalOS));

                OUT.tangentWS.xyz = TransformObjectToWorldDir(tangent);
                OUT.tangentWS.w =  1.0; 
                OUT.bitangentWS = cross(OUT.normalWS, OUT.tangentWS.xyz);
                OUT.viewDirWS = GetWorldSpaceViewDir(OUT.positionWS);
                OUT.screenPos = ComputeScreenPos(OUT.positionCS);
                return OUT;
            }

            half4 frag (Varyings IN) : SV_Target
            {
                // 在片元着色器中调用
                //float2 screenPos = i.screenPos.xy / i.screenPos.w;

                 // 修正TBN矩阵构建
                float3 normalWS = normalize(IN.normalWS);
                float3 tangentWS = normalize(IN.tangentWS.xyz);
                float3 bitangentWS = normalize(cross(normalWS, tangentWS) * IN.tangentWS.w);
                float3x3 TBN = float3x3(tangentWS, bitangentWS, normalWS);
                

                // 修正法线采样与混合
                float2 uv1 = IN.uv + _Time.y * _NormalSpeed * 0.01; // 调整速度系数
                float2 uv2 = IN.uv - _Time.y * _NormalSpeed * 0.005; // 添加差异化偏移
                
                half3 normalTS1 = UnpackNormalScale(SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, uv1), _BumpScale);
                half3 normalTS2 = UnpackNormalScale(SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, uv2), _BumpScale);
                
                // 使用正确法线混合方法
                half3 blendedNormalTS = normalize(ReorientNormal(normalTS1, normalTS2));
                
                // 转换到世界空间
                float3 pixelNormalWS = normalize(mul(blendedNormalTS, TBN));

                // 修正反射UV计算
                float2 reflectionUV = (IN.screenPos.xy / IN.screenPos.w) + pixelNormalWS.xz * (_ReflectDistort * 0.02); // 使用xz平面扰动
                reflectionUV = saturate(reflectionUV); // 防止UV越界
                
                // 使用屏幕空间采样
                float3 ReflectionColor = tex2D(_ReflectionTex, reflectionUV).rgb;

                
                // 计算屏幕UV（优化版）

                float3 viewDirWS = normalize(IN.viewDirWS);
                float2 screenUV = IN.screenPos.xy / IN.screenPos.w;

                // 获取深度值（使用URP标准方法）
                float depth = SampleSceneDepth(screenUV);
                

                float gradient = smoothstep(
                    _GradientOffset - _GradientRange, 
                    _GradientOffset + _GradientRange, 
                    IN.uv.x
                );
                

                // 使用URP内置函数重建世界坐标（推荐方式）
                float3 viewPos = ComputeViewSpacePosition(screenUV, depth, UNITY_MATRIX_P);
                float3 worldPosFromDepth = ComputeWorldSpacePosition(screenUV, depth, UNITY_MATRIX_I_VP);
                // 正确的水深计算逻辑
                float waterHeight = IN.positionWSAndFogFactor.y;
                float sceneHeight = worldPosFromDepth.y;
                
               

                float depthValue = (normalize(IN.positionWS).y - worldPosFromDepth.y)/ _DeepRange;
                depthValue = saturate(depthValue * gradient);

                float WaterDepth = (worldPosFromDepth.y - normalize(IN.positionWS).y);

                float4 DeepColorA = _DeepColor.rgba;
                float4 ShallowColorA = _ShallowColor.rgba;

                float3 DeepColor = _DeepColor.rgb;
                float3 ShallowColor = _ShallowColor.rgb;

                float DeepRange = clamp(exp((-1*depthValue) / _DeepRange),0,1);

                //DeepRange = exp(DeepRange);
                //DeepRange = clamp(DeepRange, 0.0, 1.0);

                float4 colorA = lerp(DeepColorA, ShallowColorA, DeepRange).rgba;

                float3 watercolor = lerp(DeepColor, ShallowColor, DeepRange).rgb;
                
                

                float fresnel = pow(1.0 - saturate(dot(normalWS, viewDirWS)), _FresnelPower);

                fresnel *= _FresnelIntensity;

                fresnel *= _FresnelColor.rgb;

                watercolor = lerp(watercolor, _FresnelColor.rgb, fresnel);

                float2 GrabScreenPosition = IN.screenPos.xy / IN.screenPos.w;

                float2 GrabScreenPositionUV = GrabScreenPosition + (pixelNormalWS * _UnderWaterDistort * 0.01);

                half3 UnderWaterColor = SAMPLE_TEXTURE2D(
                        _CameraOpaqueTexture, 
                        sampler_CameraOpaqueTexture, 
                        GrabScreenPositionUV
                    ).rgb;  

                    half4 UnderWaterColorA = SAMPLE_TEXTURE2D(
                        _CameraOpaqueTexture, 
                        sampler_CameraOpaqueTexture, 
                        GrabScreenPositionUV
                    ).rgba; 
                    

               

                float Reffresnel = pow(1.0 - saturate(dot(normalWS, viewDirWS)), _ReflectPower);
                Reffresnel *= _ReflectIntensity;

                ReflectionColor *=  Reffresnel;
                // 计算渐变系数（沿 X 轴）

                float2 CausticsUV = (worldPosFromDepth.xz / _CausticsScale) + _CausticsSpeed * _Time.y * 0.01; 
                float2 CausticsUV2 = -(worldPosFromDepth.xz / _CausticsScale) + _CausticsSpeed * _Time.y * 0.01;

                float3 CausticsColor01 = tex2D(_CausticsTex, CausticsUV).rgb;

                float3 CausticsColor02 = tex2D(_CausticsTex, CausticsUV2).rgb;

                CausticsColor01 = min(CausticsColor01, CausticsColor02);
                
                CausticsColor01 *= _CausticsIntensity;
                float CausticsRange = clamp(exp(-depthValue / _CausticsRange),0,1);
                CausticsColor01 *= CausticsRange;

                UnderWaterColor += CausticsColor01;

                float waterShore = clamp(exp(-depthValue / _ShoreRange),0,1);
                float shoreEdge = smoothstep(1-_ShoreEdgeWidth,1.1,waterShore) * _ShoreIntensity;

                float3 ShoreColor = _ShoreColor.rgb * UnderWaterColor;

                float4 ShoreColorA = _ShoreColor.rgba * UnderWaterColorA.rgba;

                float ShoreAlpha = ShoreColorA.a;

                float WaterOpacity = lerp(colorA.a,ShoreAlpha,0.5);

                float foamRange = 1- (smoothstep(0.3,0.5,clamp(depthValue / _FoamRange,0,1)));

                float foamMask = (smoothstep(0.3,0.5,clamp(depthValue / _FoamRange,0,1))) + 0.01;
                foamMask = 1 - (smoothstep(_FoamBlend,1,foamMask));
                float foamWidth = foamMask - _FoamWidth;

                foamRange *= _FoamFrequency;

                foamRange = sin(foamRange + _FoamSpeed * _Time.y);
                float2 NoiseUV = IN.uv * _NoiseSize;
                float3 NoiseTex = tex2D(_NoiseTexture, NoiseUV).rgb;

                foamRange += NoiseTex;
                foamRange -= _FoamDissolve;
                foamRange = step(foamWidth,foamRange);


                foamRange *= foamMask;

                float4 foamColorA = foamRange * _FoamColor.rgba;

                float3 foamColor = foamRange * _FoamColor.rgb;

                // 获取海浪高度
                float waveHeight = clamp(IN.waveHeight,0,1);

                // 根据海浪高度插值颜色
                float3 waveColor = lerp(_LowWaveColor.rgb, _HighWaveColor.rgb, saturate(waveHeight));








                float3 finalColor = lerp(waveColor + ReflectionColor + watercolor, UnderWaterColor, WaterOpacity);

                finalColor = lerp(finalColor, ShoreColor, waterShore);
                finalColor = lerp(finalColor,finalColor+foamColor,foamColorA.a);
                finalColor += saturate(shoreEdge);
                
                
                

                float alpha = _AlphaClip;
              
                //depthValue = smoothstep(0.1,1.5,1 - depthValue);

                


                half4 col = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, IN.uv);
                //return half4(color); // 转为颜色输出
                return half4(finalColor,_AlphaClip);
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }
            ZWrite [_ZWrite]
            ColorMask 0
            Cull [_Cull]

            HLSLPROGRAM

            #pragma mulit_compile_instancing
            #pragma mulit_compile_DOTS_INSTANCING_ON

            #pragma vertex vert
            #pragma fragment fragDepth

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
            };

            float _AlphaClip;

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }

            float4 fragDepth(Varyings IN) : SV_Target
            {
                clip(1.0 - _AlphaClip);
                return 0;
            }
            ENDHLSL
        }

        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode" = "DepthNormals" }
            ZWrite [_ZWrite]
            Cull [_Cull]

            HLSLPROGRAM

            #pragma mulit_compile_instancing
            #pragma mulit_compile_DOTS_INSTANCING_ON

            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float4 tangentOS : TANGENT;
                float3 normalOS : NORMAL;
                float2 texcoord : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float4 tangentWS : TEXCOORD2;
            };

            float _AlphaClip;

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;

                output.uv = input.texcoord;
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS, input.tangentOS);

                float3 viewDirWS = GetWorldSpaceNormalizeViewDir(vertexInput.positionWS);                           
                output.normalWS = half3(normalInput.normalWS);
                float sign = input.tangentOS.w * float3(GetOddNegativeScale());
                output.tangentWS = half4(normalInput.tangentWS.xyz,sign);

                return output;               
            }

            half4 frag(Varyings input) : SV_Target
            {
                clip(1.0 - _AlphaClip);
                float3 normalWS = input.normalWS.xyz;
                // //float2 uv = i.uv;
                

                // float3 NormalDirWS = normalWs;

                // float3 NormalDirWS2 = normalWs * 2;

                // float2 normalOffset1 = uv + _Time.y * _NormalSpeed * 0.01;
                // float2 normalOffset2 = uv - _Time.y * _NormalSpeed * 0.005;


                //  // 正确采样法线贴图（返回float4）
                // half4 normalMap1 = SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, normalOffset1);
                // half4 normalMap2 = SAMPLE_TEXTURE2D(_NormalMap, sampler_NormalMap, normalOffset2);

                // half3 normalTS1 = UnpackNormalScale(normalMap1, _BumpScale);
                // half3 normalTS2 = UnpackNormalScale(normalMap2, _BumpScale);

                // // 使用URP内置混合函数
                // half3 blendedNormalTS = ReorientNormal(normalTS1, normalTS2);

                // float3 pixelNormalWS = normalize(mul(blendedNormalTS, TBN));

                // float4 reflection = tex2D(_ReflectionTex, i.uv);

                // float2 ReflectionTexUV = i.screenPos.xy / i.screenPos.w +  pixelNormalWS.xy * (_ReflectDistort * 0.01);  
                
                // float3 ReflectionColor = tex2D( _ReflectionTex,ReflectionTexUV).rgb;

                //float3 depth = SampleSceneDepth(i.uv);
                 //float depthValue = saturate((waterHeight - sceneHeight) / _DeepRange); // 添加可调参数


                //float depthValue = Linear01Depth(depth, _ZBufferParams) + 0.5 * 2;
                //depthValue = saturate(depthValue * gradient);

                return half4(NormalizeNormalPerPixel(normalWS),0.0);
            }
            ENDHLSL
            
        }
    }
}