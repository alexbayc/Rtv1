/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   create_cfloat3x.c                                  :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: odale-dr <odale-dr@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/07/13 16:29:01 by odale-dr          #+#    #+#             */
/*   Updated: 2019/07/13 16:29:54 by odale-dr         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "rtv1.h"

cl_float3 create_cfloat3x (float x, float y, float z)
{
	cl_float3 re;

	re.x = x;
	re.y = y;
	re.z = z;

	return re;
}