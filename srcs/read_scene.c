/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   read_scene.c                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: odale-dr <odale-dr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/07/11 18:17:04 by odale-dr          #+#    #+#             */
/*   Updated: 2019/07/18 21:27:20 by odale-dr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "rtv1.h"

void			pars(t_obj * cpu_spheres, char **x, int r)
{
	static int	i;
	int			t;

	t = i;
	while (t < r && x[t] != NULL)
	{
		// ft_putstr(ft_itoa(t));
		// ft_putstr("         =           ");
		// ft_putstr(x[t]);
		// ft_putstr("\n");
		if ((ft_strstr(x[t], "Type") != NULL) &&(ft_strstr(x[t], "//") == NULL))
			type(cpu_spheres, x, r, t);
		t++;
	}
	i = r + 1;
}

void			read_sc(t_obj* cpu_spheres, int fd)
{
	char		**x;
	int			i;

	i = 0;
	x = malloc(sizeof(char *) * 500);
	while (get_next_line(fd, &(x[i])) != 0)
	{
		if (ft_strlen(x[i]) == 0)
		{
			pars(cpu_spheres, x, i);
			// ft_putstr("\n");
			//break ;
		}
		i++;
	}
	pars(cpu_spheres, x, i);
	// ft_putstr("\n------------");
	// ft_putstr(ft_itoa(i));

	free(x);
}

void				read_scene(t_obj * cpu_spheres, char *s)
{
	int			fd;

	fd = 0;
	if (s == NULL)
		return ;
	fd = open(s, O_RDONLY);
	if (fd == 0)
		return ;
	read_sc(cpu_spheres, fd);
	close(fd);
}