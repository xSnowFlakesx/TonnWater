Shader "Hidden/KawaseBlur" {
    Properties { _MainTex ("", 2D) = "white" {} }
    SubShader {
        Pass {

            Cull Off ZWrite Off ZTest Always
            HLSLPROGRAM
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #pragma vertex vert
            #pragma fragment frag

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };


            struct v2f { float2 uv : TEXCOORD0; float4 pos : SV_POSITION; };


            v2f vert(appdata v) { 
                v2f o; 
                o.pos = TransformObjectToHClip(v.vertex.xyz);
                o.uv = v.uv;
                return o;
            }


            sampler2D _MainTex;
            uniform half _Offset;
            float4 _MainTex_TexelSize;


            half4 frag(v2f i) : SV_Target {
				half4 o = 0;
				o += tex2D(_MainTex, i.uv + float2(_Offset + 0.5, _Offset + 0.5) * _MainTex_TexelSize.xy);
				o += tex2D(_MainTex, i.uv + float2(-_Offset - 0.5, _Offset + 0.5) * _MainTex_TexelSize.xy);
				o += tex2D(_MainTex, i.uv + float2(-_Offset - 0.5, -_Offset - 0.5) * _MainTex_TexelSize.xy);
				o += tex2D(_MainTex, i.uv + float2(_Offset + 0.5, -_Offset - 0.5) * _MainTex_TexelSize.xy);
				return o * 0.25;
			}
			ENDHLSL
		}
	}
}


