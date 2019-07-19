/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   type.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: odale-dr <odale-dr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/07/13 16:33:59 by odale-dr          #+#    #+#             */
/*   Updated: 2019/07/18 21:27:57 by odale-dr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "rtv1.h"

void				plane(t_obj *cpu_spheres, char **x, int i, int t)
{
	float			q;
	float			y;
	float			z;

	cpu_spheres[i].type = PLANE;
	q = return_float(x[t + 4]);
	y = return_float(x[t + 5]);
	cpu_spheres[i].position = create_cfloat3x(q, y, return_float(x[t + 6]));
	cpu_spheres[i].radius = return_float(x[t + 1]);
	q = return_float(x[t + 10]);
	y = return_float(x[t + 11]);
	cpu_spheres[i].color = create_cfloat3x(q, y, return_float(x[t + 12]));
	q = return_float(x[t + 16]);
	y = return_float(x[t + 17]);
	cpu_spheres[i].emission = create_cfloat3x(q, y, return_float(x[t + 18]));
	cpu_spheres[i].reflection = return_float(x[t + 20]);
	cpu_spheres[i].plane_d = return_float(x[t + 21]);
	// ft_putstr("\n ViCHEL");
}

void				cyl(t_obj *cpu_spheres, char **x, int i, int t)
{
	float			q;
	float			y;
	float			z;

	cpu_spheres[i].type = CYLINDER;
	cpu_spheres[i].radius = return_float(x[t + 1]);
	q = return_float(x[t + 4]);
	y = return_float(x[t + 5]);
	cpu_spheres[i].v = create_cfloat3x(q, y, return_float(x[t + 6]));
	q = return_float(x[t + 10]);
	y = return_float(x[t + 11]);
	cpu_spheres[i].position = create_cfloat3x(q, y, return_float(x[t + 12]));
	q = return_float(x[t + 16]);
	y = return_float(x[t + 17]);
	cpu_spheres[i].color = create_cfloat3x(q, y, return_float(x[t + 18]));
	q = return_float(x[t + 22]);
	y = return_float(x[t + 23]);
	cpu_spheres[i].emission = create_cfloat3x(q, y, return_float(x[t + 24]));
	cpu_spheres[i].reflection = return_float(x[t + 26]);
	// ft_putstr("\n ViCHEL");
}

void				cone(t_obj *cpu_spheres, char **x, int i, int t)
{
	float			q;
	float			y;
	float			z;

	cpu_spheres[i].type = CONE;
	cpu_spheres[i].radius = return_float(x[t + 1]);
	q = return_float(x[t + 4]);
	y = return_float(x[t + 5]);
	cpu_spheres[i].v = create_cfloat3x(q, y, return_float(x[t + 6]));
	q = return_float(x[t + 10]);
	y = return_float(x[t + 11]);
	cpu_spheres[i].position = create_cfloat3x(q, y, return_float(x[t + 12]));
	q = return_float(x[t + 16]);
	y = return_float(x[t + 17]);
	cpu_spheres[i].color = create_cfloat3x(q, y, return_float(x[t + 18]));
	q = return_float(x[t + 22]);
	y = return_float(x[t + 23]);
	cpu_spheres[i].emission = create_cfloat3x(q, y, return_float(x[t + 24]));
	cpu_spheres[i].reflection = return_float(x[t + 26]);
	// ft_putstr("\n ViCHEL");
}

void				sp(t_obj *cpu_spheres, char **x, int i, int t)
{
	float			q;
	float			y;
	float			z;

	cpu_spheres[i].type = SPHERE;
	q = return_float(x[t + 4]);
	y = return_float(x[t + 5]);
	cpu_spheres[i].position = create_cfloat3x(q, y, return_float(x[t + 6]));
	cpu_spheres[i].radius = return_float(x[t + 1]);
	q = return_float(x[t + 10]);
	y = return_float(x[t + 11]);
	cpu_spheres[i].color = create_cfloat3x(q, y, return_float(x[t + 12]));
	q = return_float(x[t + 16]);
	y = return_float(x[t + 17]);
	cpu_spheres[i].emission = create_cfloat3x(q, y, return_float(x[t + 18]));
	cpu_spheres[i].reflection = return_float(x[t + 20]);
	// ft_putstr("\n ViCHEL");
}

void				type(t_obj *cpu_spheres, char **x, int r, int t)
{
	static int		i;

	if (i == 0)
		i = 0;
	if (ft_strstr(x[t], "sph"))
		sp(cpu_spheres, x, i, t);
	if (ft_strstr(x[t], "cone"))
		cone(cpu_spheres, x, i, t);
	if (ft_strstr(x[t], "cyl"))
		cyl(cpu_spheres, x, i, t);
	if (ft_strstr(x[t], "pla"))
		plane(cpu_spheres, x, i, t);
	i++;
}
