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

void main()
{
  vec3 normal = normalize(v_normal);
  mat4 world = u_view * u_model;

  vec3 surfaceToLight;
  float lightDist;
  surfaceToLight = camera_position.xyz - v_position.xyz;
  lightDist = 1;
  // always shine towards camera

  // Calculate the cosine of the angle of incidence (brightness)
  //float brightness = dot(normal, normalize(surfaceToLight)) / lightDist;
  float nc = dot(normal, normalize(surfaceToLight)); // use for selecting texture
  float brightness = clamp(nc, 0, 1);
  float ambInt = 0.1;

  brightness = clamp(brightness+ambInt, 0, 1.0);
  //brightness = 1.0;

  vec4 clr1;
  clr1 = v_color;
  //fragColor =  clr1* brightness * vec4(u_light_intensity, 1);
  fragColor = clr1 * brightness;
  // force alpha
  fragColor.a = 1.0;

  //fragColor = v_color;
  //fragColor.g = clamp(v_color.g * t_color.g, 0, 1);
  //fragColor = v_color * t_color;
  //fragColor = v_color*v_green;
  //fragColor = vec4( 1.0, 0.0, 1.0, 1.0 );
}
