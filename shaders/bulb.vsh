#version 330
layout(location = 0) in vec4 a_position;
layout(location = 1) in vec2 a_texcoord;
layout(location = 2) in vec3 a_normal;
layout(location = 3) in vec4 a_color;


uniform mat4 u_model;
uniform int bulb_index;

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

out vec4 v_position;
out vec3 v_normal;
out vec4 v_color;

out vec4 v_red;
out vec4 v_green;
out vec4 v_blue;

void main()
{
  //v_normal = a_normal;
  v_normal = vec3(u_model * vec4(a_normal, 0));
  v_position = a_position;
  v_color = light_diffuse[bulb_index];

  v_red = camera_position;
  v_green = main_light_position;
  v_blue = main_light_diffuse;

  gl_Position = u_projection * u_view * u_model * a_position;
}
