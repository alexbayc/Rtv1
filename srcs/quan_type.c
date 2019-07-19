/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   quan_type.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: odale-dr <odale-dr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/07/13 14:12:15 by odale-dr          #+#    #+#             */
/*   Updated: 2019/07/18 21:25:07 by odale-dr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "rtv1.h"

int				quan_type(char *s)
{
	int			quan;
	int			fd;
	char		**f;

	f = malloc(sizeof(char *));
	quan = 0;
	fd = open(s, O_RDONLY);
	if (fd == 0)
		return(0);
	while (get_next_line(fd, &(f[0])) != 0)
	{
		if (ft_strstr(f[0], "Type") != NULL  && ft_strstr(f[0], "//") == NULL)
			quan++;
	}
	close(fd);
	free(f);
	// ft_putstr("\n ______________________________");
	// ft_putstr(ft_itoa(quan));
	// ft_putstr("\n");
	return(quan);
}