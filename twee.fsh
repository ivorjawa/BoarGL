#version 330
uniform mat4 u_model;
//uniform mat4 u_view;
//uniform mat4 u_normal;
layout (std140) uniform shader_data
{
  vec4 camera_position;
  vec4 main_light_position;
  vec4 main_light_diffuse;

  mat4 u_view;
  mat4 u_projection;

  vec4 light_positions[30];
  vec4 light_diffuse[30];
};


uniform vec3 u_light_intensity;
uniform vec3 u_light_position;

in vec4 v_position;
in vec3 v_normal;
in vec4 v_color;

in vec4 v_red;
in vec4 v_green;
in vec4 v_blue;

uniform sampler2D tex0;

out vec4 fragColor;

// default bulb radius = .03
void main()
{
  vec3 normal = normalize(v_normal);
  mat4 world = u_view * u_model;
  vec4 t_color = vec4( 0.0, 0.0, 0.0, 1.0 );

  bool do_glow = false;
  vec4 glow_color = vec4(0.0, 0.0, 0.0, 1.0);
  float light_sort = .03 * 4;
  vec4 spos = world * v_position;
  for(int i = 0; i< 30; i++){
    vec4 lp = light_positions[i];
    vec4 lpos = world * lp;
    vec3 surfaceToLight = lpos.xyz - spos.xyz;
    float lightDist = length(surfaceToLight);
    float brightness = clamp(dot(normal, normalize(surfaceToLight)), 0, 1);
    float weight = 1.0 / (1.0 + pow(lightDist*5, 2) );
    vec4 crap = (weight*2.0) * light_diffuse[i];
    t_color = t_color + crap;
    if((lightDist < light_sort)){
           do_glow = true;
           //light_sort = lightDist;
           float gweight = 1.0 / (1.0 + pow(lightDist*20, 2) );
           glow_color = light_diffuse[i]*gweight;
           glow_color.a = 1.0;
           //glow_color = vec4(1.0, 1.0, 0.0, 1.0);
    }
  }

  //fragColor = v_color;
  //fragColor.g = v_color.g * t_color.g;
  float bwite = (t_color.r + t_color.g + t_color.b) / 3;
  if(do_glow){
    fragColor = v_color*bwite+glow_color;
    fragColor.a = 1.0;
    //fragColor.g = clamp(v_color.g * t_color.g, .2, 1);
  } else {
    fragColor = v_color * bwite;
    //fragColor.g = bwite;
    fragColor.a = 1.0;
    //fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    //fragColor.g = clamp(v_color.g * t_color.g, .2, 1);
  }
  fragColor = clamp(fragColor, 0, 1);
}
