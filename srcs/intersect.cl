typedef	struct 	s_vec3 t_vec3;

struct 	s_vec3
{
	float x;
	float y;
	float z;
	float w;
};

typedef struct 	s_mat4
{
	double matrix[4][4];
} t_mat4;

typedef struct s_ray
{
	t_vec3 orig;
	t_vec3 dir;
	t_vec3 hit;
	double t;

} t_ray;

typedef struct	Material {
	t_vec3 diffuse_color;
	t_vec3 albendo;
	float specular_exponent;
	float refractive_index;
}				t_material;

typedef struct s_light {
	t_vec3 position;
	float intensity;
} t_light;

typedef struct s_figure
{
	t_vec3		center;
	t_material	material;
	float		radius;
	t_vec3		v;
	double		angle;
	int			type;
	int			min;
	int			max;
}				t_figure;

enum e_figure {PLANE, SPHERE, CYLINDER, CONE};

double		sphere_intersection(t_figure *figure, t_ray *ray, float *t0);
double		cone_intersection(t_figure *object, t_ray *ray, float *t0);
double		cylinder_intersection(t_figure *object, t_ray *ray, float *t0);
double		plane_intersection(t_figure *object, t_ray *ray, float *t0);
int			have_solutions(double d);
double		get_solution(double a, double b, double c, float *t0);

t_vec3		reflect(t_vec3 I, t_vec3 n);
t_vec3		refract(t_vec3 I, t_vec3 N, const float eta_t, const float eta_i);

t_vec3		ft_vec3_scalar_multiply(t_vec3 a, float b);
float		ft_vec3_dot_multiply(t_vec3 a, t_vec3 b);
t_vec3		ft_vec3_substract(t_vec3 a, t_vec3 b);
float		ft_vec3_multiply_cone(t_vec3 a, t_vec3 b);
t_vec3		ft_vec3_create(float x, float y, float z);
t_vec3		ft_vec3_multiply_matrix(t_vec3 v, t_mat4 m);
t_mat4		ft_mat4_multiply_mat4(t_mat4 a, t_mat4 b);
t_mat4		ft_mat4_translation_matrix(t_vec3 v);
t_mat4		ft_mat4_rotation_matrix(t_vec3 axis, double alpha);
t_mat4		ft_mat4_identity_matrix(void);
t_vec3		ft_vec3_normalize(t_vec3 vect);
float 		ft_vec3_norm(t_vec3 vect);
t_vec3		ft_vec3_sum(t_vec3 a, t_vec3 b);
t_vec3		ft_vec3_neg(t_vec3 v);

t_vec3		plane_get_normal(t_ray *ray, t_figure *figure);
t_vec3		cone_get_normal(t_ray *ray, t_figure *figure);
t_vec3		sphere_get_normal(t_ray *ray, t_figure *figure);
t_vec3		cylinder_get_normal(t_ray *ray, t_figure *figure);

int			is_any_figure_closer(double *closest, double cache);
int			scene_intersect(__constant t_figure *objs, int obj_num, t_ray *ray, t_vec3 *hit, t_vec3 *N, t_material *material);
t_vec3		cast_ray(__constant t_figure *objs, __constant t_light *lights, int obj_num, int lights_count, t_ray *ray, size_t depth);

//====================================//

__kernel void init_calculations(
	__global t_vec3 *vecs, 
	__constant t_figure *objects,
	__constant t_light *lights,
	int objs_count,
	int lights_count, //TODO use const
	__global t_vec3 *out_vecs,
	const unsigned int count)
{
	t_vec3 origin, dir;
	float xa = 0, ya = 0, za = 0;
	float eyex = 0, eyey = 0, eyez = 0;
	int id = get_global_id(0);
	if (id < count)
	{
		origin = ft_vec3_create(eyex, eyey, eyez);
		dir = ft_vec3_multiply_matrix(vecs[id], ft_mat4_rotation_matrix((t_vec3) {0,-1,0}, xa));
		barrier(CLK_LOCAL_MEM_FENCE);
		out_vecs[id] = cast_ray(objects, lights, objs_count, lights_count, &(t_ray){origin, dir}, 0);
	}
}

float ft_vec3_norm(t_vec3 vect)
{
	return (sqrt(vect.x * vect.x + vect.y * vect.y + vect.z * vect.z));
}

float ft_vec3_dot_multiply(t_vec3 a, t_vec3 b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

t_vec3 ft_vec3_neg(t_vec3 v)
{
	return (ft_vec3_scalar_multiply(v, -1));
}

t_vec3 ft_vec3_create(float x, float y, float z)
{
	t_vec3 new;

	new.x = x;
	new.y = y;
	new.z = z;
	return (new);
}

t_vec3 ft_vec3_normalize(t_vec3 vect)
{
	t_vec3 res = vect;
	float norm = ft_vec3_norm(res);
	res.x = vect.x / norm;
	res.y = vect.y / norm;
	res.z = vect.z / norm;
	return (res);
}

t_mat4	ft_mat4_identity_matrix(void)
{
	t_mat4	res;
	int				i;
	int				j;

	i = -1;
	while (++i < 4)
	{
		j = -1;
		while (++j < 4)
			res.matrix[i][j] = 0;
	}
	res.matrix[0][0] = 1;
	res.matrix[1][1] = 1;
	res.matrix[2][2] = 1;
	res.matrix[3][3] = 1;
	return (res);
}

t_vec3	ft_vec3_multiply_matrix(t_vec3 v, t_mat4 m)
{
	t_vec3	res;

	res.x = v.x * m.matrix[0][0] +
			v.y * m.matrix[0][1] +
			v.z * m.matrix[0][2] +
			v.w * m.matrix[0][3];
	res.y = v.x * m.matrix[1][0] +
			v.y * m.matrix[1][1] +
			v.z * m.matrix[1][2] +
			v.w * m.matrix[1][3];
	res.z = v.x * m.matrix[2][0] +
			v.y * m.matrix[2][1] +
			v.z * m.matrix[2][2] +
			v.w * m.matrix[2][3];
	res.w = v.x * m.matrix[3][0] +
			v.y * m.matrix[3][1] +
			v.z * m.matrix[3][2] +
			v.w * m.matrix[3][3];
	return (res);
}

t_mat4	ft_mat4_multiply_mat4(t_mat4 a, t_mat4 b)
{
	t_mat4	res;
	int				i;
	int				j;

	i = -1;
	while (++i < 4)
	{
		j = -1;
		while (++j < 4)
		{
			res.matrix[i][j] = a.matrix[i][0] * b.matrix[0][j] +
				a.matrix[i][1] * b.matrix[1][j] +
				a.matrix[i][2] * b.matrix[2][j] +
				a.matrix[i][3] * b.matrix[3][j];
		}
	}
	return (res);
}

t_mat4	ft_mat4_translation_matrix(t_vec3 v)
{
	t_mat4	res;

	res = ft_mat4_identity_matrix();
	res.matrix[0][3] = v.x;
	res.matrix[1][3] = v.y;
	res.matrix[2][3] = v.z;
	return (res);
}

t_mat4	ft_mat4_rotation_matrix(t_vec3 axis, double alpha)
{
	t_mat4			res;
	double			sinus;
	double			cosin;
	double			inv_cosin;

	res = ft_mat4_identity_matrix();
	axis = ft_vec3_normalize(axis);
	sinus = sin(alpha);
	cosin = cos(alpha);
	inv_cosin = 1 - cosin;
	res.matrix[0][0] = cosin + inv_cosin * axis.x * axis.x;
	res.matrix[1][0] = inv_cosin * axis.x * axis.y - sinus * axis.z;
	res.matrix[2][0] = inv_cosin * axis.x * axis.z + sinus * axis.y;
	res.matrix[0][1] = inv_cosin * axis.y * axis.x + sinus * axis.z;
	res.matrix[1][1] = cosin + inv_cosin * axis.y * axis.y;
	res.matrix[2][1] = inv_cosin * axis.y * axis.z - sinus * axis.x;
	res.matrix[0][2] = inv_cosin * axis.z * axis.x - sinus * axis.y;
	res.matrix[1][2] = inv_cosin * axis.z * axis.y + sinus * axis.x;
	res.matrix[2][2] = cosin + inv_cosin * axis.z * axis.z;
	return (res);
}

int	have_solutions(double d)
{
	if (d > 0)
		return (2);
	else if (d == 0)
		return (1);
	else
		return (0);
}

double		get_solution(double a, double b, double c, float *t0)
{
	double		d;
	double		tmp1;
	double		tmp2;

	d = b * b - 4.0 * a * c;
	if (have_solutions(d) == 0)
		return (0);
	else if (have_solutions(d) == 1)
		*t0 = - b / (2 * a);
	else
	{
		tmp1 = sqrt(d);
		tmp2 = 1 / (2 * a);
		if (((*t0 = (- b - tmp1) * tmp2)) < 0.003)
			if ((*t0 = ((- b + tmp1) * tmp2)) < 0.003)
				return (0);
	}
		return (1);
}

t_vec3	ft_vec3_substract(t_vec3 a, t_vec3 b)
{
	t_vec3 new;

	new.x = (a.x - b.x);
	new.y = (a.y - b.y);
	new.z = (a.z - b.z);
	return (new);
}

t_vec3	ft_vec3_sum(t_vec3 a, t_vec3 b)
{
	t_vec3 new;

	new.x = (a.x + b.x);
	new.y = (a.y + b.y);
	new.z = (a.z + b.z);
	return (new);
}

float ft_vec3_multiply_cone(t_vec3 a, t_vec3 b)
{
	return (a.x * b.x - a.y * b.y + a.z * b.z);
}

t_vec3 ft_vec3_scalar_multiply(t_vec3 a, float b)
{
	return ((t_vec3){a.x * b, a.y * b, a.z * b});
}
/*--------------------intersection------------------------- */

t_vec3	reflect(t_vec3 I, t_vec3 n)
{
	t_vec3 temp;
	
	temp = ft_vec3_scalar_multiply(n, 2.f * ft_vec3_dot_multiply(I, n));
	return ft_vec3_substract(I, temp);
}

t_vec3 refract(t_vec3 I, t_vec3 N, const float eta_t, const float eta_i)
{ // Snell's law
	float cosi = -max(-1.f, min(1.f, ft_vec3_dot_multiply(I,N)));
    if (cosi<0) 
		return (t_vec3){1,0,0,1};//refract(I, ft_vec3_neg(N), eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta*eta*(1 - cosi*cosi);
    return k<0 ? (t_vec3){1,0,0,1}
				: ft_vec3_sum(ft_vec3_scalar_multiply(I,eta), ft_vec3_scalar_multiply(N,(eta*cosi - sqrtf(k)))); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}


double		sphere_intersection(t_figure *object, t_ray *ray, float *t0)
{
	t_vec3 temp = ft_vec3_substract(ray->orig, object->center);
	double a = ft_vec3_dot_multiply(ray->dir, ray->dir);
	double b = ft_vec3_dot_multiply(ft_vec3_scalar_multiply(temp, 2), ray->dir);
	double c = ft_vec3_dot_multiply(temp, temp) - object->radius * object->radius;
	return (get_solution(a, b, c, t0));
}

double	cone_intersection(t_figure *object, t_ray *ray, float *t0)
{
	t_vec3	x;
	double	a;
	double	b;
	double	c;

	x = ft_vec3_substract(ray->orig, object->center);
	a = ft_vec3_dot_multiply(ray->dir, object->v);
	a = ft_vec3_dot_multiply(ray->dir, ray->dir) - (1 + object->radius * object->radius) * a * a;
	b = 2.0 * (ft_vec3_dot_multiply(ray->dir, x) - (1 + object->radius * object->radius)
		* ft_vec3_dot_multiply(ray->dir, object->v) * ft_vec3_dot_multiply(x, object->v));
	c = ft_vec3_dot_multiply(x, object->v);
	c = ft_vec3_dot_multiply(x, x) - (1 + object->radius * object->radius) * c * c;
	return (get_solution(a, b, c, t0));
}

double		plane_intersection(t_figure *object, t_ray *ray, float *t0)
{
	double tmp;

	tmp = object->center.x * ray->dir.x + object->center.y * ray->dir.y + object->center.z * ray->dir.z;
	if (!tmp)
		return (0);
	*t0 = -(object->center.x * ray->orig.x +  object->center.y * ray->orig.y +  object->center.z * ray->orig.z +  object->center.w) / tmp;
	return ((*t0 >= 0.0003) ? 1 : 0);
}

double		cylinder_intersection(t_figure *object, t_ray *ray, float *t0)
{
	t_vec3	x;
	double	a;
	double	b;
	double	c;

	x = ft_vec3_substract(ray->orig, object->center);
	a = ft_vec3_dot_multiply(ray->dir, object->v);
	a = ft_vec3_dot_multiply(ray->dir, ray->dir) - a * a;
	b = 2 * (ft_vec3_dot_multiply(ray->dir, x) - ft_vec3_dot_multiply(ray->dir, object->v)
		* ft_vec3_dot_multiply(x, object->v));
	c = ft_vec3_dot_multiply(x, object->v);
	c = ft_vec3_dot_multiply(x, x) - c * c - object->radius * object->radius;
	return (get_solution(a, b, c, t0));
}


/* ********************************** */

int	is_any_figure_closer(double *closest, double cache)
{
	if (cache > 0 && cache < *closest)
	{
		*closest = cache;
		return (1);
	}
	return (0);
}

t_vec3 sphere_get_normal(t_ray *ray, t_figure *figure)
{
	return ft_vec3_substract(ray->hit, figure->center);
}

t_vec3 plane_get_normal(t_ray *ray, t_figure *figure)
{
	if (ft_vec3_dot_multiply(ray->dir, figure->v) < 0)
		return (figure->v);
	return ft_vec3_scalar_multiply(figure->v, -1);
}

t_vec3 cylinder_get_normal(t_ray *ray, t_figure *figure)
{
	double	m;
	t_vec3	n;
	t_vec3	p;
	
	m = ft_vec3_dot_multiply(ray->dir, figure->v) * ray->t
		+ ft_vec3_dot_multiply(ft_vec3_substract(ray->orig, figure->center), figure->v);
	p = ft_vec3_sum(ray->orig, ft_vec3_scalar_multiply(ray->dir, ray->t));
	n = ft_vec3_normalize(ft_vec3_substract(ft_vec3_substract(p, figure->center), ft_vec3_scalar_multiply(figure->v, m)));
	if (ft_vec3_dot_multiply(ray->dir, n) > 0.0001)
		n = ft_vec3_scalar_multiply(n, -1);
	return n;
}

t_vec3 cone_get_normal(t_ray *ray, t_figure *figure)
{
	t_vec3	n;
	double	m;
	m = ft_vec3_dot_multiply(ray->dir, figure->v) * ray->t
		+ ft_vec3_dot_multiply(ft_vec3_substract(ray->orig, figure->center), figure->v);
	n = ft_vec3_scalar_multiply(ft_vec3_scalar_multiply(figure->v, m), (1 + figure->radius * figure->radius));
	n = ft_vec3_normalize(ft_vec3_substract(ft_vec3_substract(ray->hit, figure->center), n));
	if (ft_vec3_dot_multiply(ray->dir, n) > 0.0001)
		n = ft_vec3_scalar_multiply(n, -1);
	return n;
}

int scene_intersect(__constant t_figure *figures, int obj_num, t_ray *ray, t_vec3 *hit, t_vec3 *N, t_material *material)
{
 	double closest = FLT_MAX; 
	float dist_i;
	float object_dist = FLT_MAX; 
	int i = 0;
	double (*intersection[5])() = {plane_intersection, sphere_intersection, cylinder_intersection, cone_intersection};//TODO think about it
	t_vec3 (*get_normal[5])() = {plane_get_normal, sphere_get_normal, cylinder_get_normal, cone_get_normal};
	while (i < obj_num)
	{
		if (intersection[figures[i].type](&figures[i], ray, &dist_i) && dist_i < object_dist)
		{
			is_any_figure_closer(&closest, dist_i); 
			object_dist = dist_i;
			ray->t = dist_i;
			t_vec3 temp = ft_vec3_scalar_multiply(ray->dir, dist_i);
			*hit = ft_vec3_sum(ray->orig, temp);
			ray->hit = *hit;
			temp = get_normal[figures[i].type](ray, &figures[i]); // problem also cause of shading not working
			*N = ft_vec3_normalize(temp);
			*material = figures[i].material;
		}
		i++;
	}
	return closest < 1000;
}

t_vec3 cast_ray(__constant t_figure *objs, __constant t_light *lights, int obj_num, int elum_num, t_ray *ray, size_t depth)
{//TODO rewrite without recursion, think about 'for' loop
	t_vec3 point;
	t_vec3 N;
	t_material material; 
	int	i;
	float sphere_dist = FLT_MAX;

	barrier(CLK_LOCAL_MEM_FENCE);
	if( depth > 4 || !scene_intersect(objs, obj_num, ray, &point, &N, &material))
		return ft_vec3_create(0.2, 0.7, 0.8); // background color

	// t_ray reflect_ray;
	// reflect_ray.dir = ft_vec3_normalize(reflect(ray->dir, N));
	// reflect_ray.orig  = ft_vec3_dot_multiply(reflect_ray.dir, N) < 0 
	// 					? ft_vec3_substract(point, ft_vec3_scalar_multiply(N, 1e-3)) 
	// 					: ft_vec3_sum(point, ft_vec3_scalar_multiply(N, 1e-3));
	// t_vec3 reflect_color = cast_ray(game, &reflect_ray, depth + 1);
	// t_ray refract_ray;
	// refract_ray.dir = ft_vec3_normalize(refract(ray->dir, N, material.refractive_index, 1.0f));
	// refract_ray.orig = ft_vec3_dot_multiply(refract_ray.dir, N) < 0 
	// 					? ft_vec3_substract(point, ft_vec3_scalar_multiply(N, 1e-3)) 
	// 					: ft_vec3_sum(point, ft_vec3_scalar_multiply(N, 1e-3));
	// t_vec3 refract_color = cast_ray(game, &refract_ray, depth + 1);
	
	float diffuse_light_intensity = 0;
	float specular_light_intensity = 0;
	i = -1;

	while (++i < elum_num)
	{
		t_vec3	light_pos = lights[i].position;
		float	light_intensity = lights[i].intensity;

		t_vec3 light_dir = ft_vec3_normalize(ft_vec3_substract(light_pos, point));
		double light_distance = ft_vec3_norm(ft_vec3_substract(light_pos, point));
		t_ray shadow_ray;
		shadow_ray.orig = (ft_vec3_dot_multiply(light_dir, N) < 0)
			? ft_vec3_substract(point, ft_vec3_scalar_multiply(N, 1e-3))
			: ft_vec3_sum(point, ft_vec3_scalar_multiply(N, 1e-3));
		shadow_ray.dir = light_dir;
		t_vec3 shadow_pt, shadow_N;
		t_material temp_material;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (scene_intersect(objs, obj_num, &shadow_ray, &shadow_pt, &shadow_N, &temp_material) 
			&& (ft_vec3_norm(ft_vec3_substract(shadow_pt, shadow_ray.orig)) < light_distance))
			continue;
		diffuse_light_intensity += light_intensity
			* max(0.0f, ft_vec3_dot_multiply(ft_vec3_normalize(light_dir), ft_vec3_normalize(N)));
		specular_light_intensity += powf(max(0.f, ft_vec3_dot_multiply(ft_vec3_scalar_multiply(
			reflect(ft_vec3_scalar_multiply(light_dir, -1), N), -1), ray->dir)),
		 	material.specular_exponent)*light_intensity;
	}
		
	// return ft_vec3_sum(ft_vec3_sum(ft_vec3_sum(ft_vec3_scalar_multiply(material.diffuse_color,\
	//  diffuse_light_intensity * material.albendo.x), \
	//  	 ft_vec3_scalar_multiply((t_vec3){1,1,1}, specular_light_intensity *  material.albendo.y)),\
	// 	 ft_vec3_scalar_multiply(reflect_color,  material.albendo.z)), ft_vec3_scalar_multiply(refract_color,  material.albendo.w));					//ft_vec3_scalar_multiply(&material.diffuse_color, diffuse_light_intensity);
	return ft_vec3_sum(ft_vec3_scalar_multiply(material.diffuse_color, diffuse_light_intensity * material.albendo.x), \
		ft_vec3_scalar_multiply((t_vec3){1,1,1}, specular_light_intensity *  material.albendo.y));
}