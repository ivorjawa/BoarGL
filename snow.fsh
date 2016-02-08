#version 330
uniform mat4 u_model;
uniform mat4 u_normal;

in vec4 v_position;
in vec3 v_normal;
in vec4 v_color;

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


uniform sampler2D flake_tex;

out vec4 fragColor;

void main()
{
  vec4 t_color;
  t_color = vec4( 0.0, 0.0, 0.0, 0.0 ); // fix this crap for transparency
  mat4 world = u_view * u_model;
  //mat4 world = u_view * u_model * u_extralight_rot;
  vec4 spos = world * v_position;
  for(int i = 0; i < 30; i++){
    vec4 lp = light_positions[i];
    vec4 lpos = world * lp;
    float lightDist = length(lpos.xyz - spos.xyz);
    float weight = 1.0 / (1.0 + pow(lightDist*2, 2) );
    vec4 crap = weight/2 * light_diffuse[i];
    t_color = t_color + crap;
    //t_color = t_color + weight * light_diffuse[i];
    //t_color = light_diffuse[5];
  }
  vec4 tex_color =  texture(flake_tex, gl_PointCoord);
  t_color = tex_color * t_color;
  t_color = clamp(t_color, 0, 1.0);
  fragColor = t_color;
  float bwite = (t_color.r + t_color.g + t_color.b) / 3;
  //fragColor.a *= bwite;

  //fragColor = tex_color;
  //fragColor = v_color*v_green;
  //fragColor = vec4( 1.0, 0.0, 1.0, 1.0 );
}
