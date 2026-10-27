/*
  Copyright (C) 2017 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include </fs2/home/liuzhonglan/wys/20250105/rift/include/comp/lithosphere_rift.h>
#include </fs2/home/liuzhonglan/wys/20250105/rift/include/temp/lithosphere_rift.h>
#include </fs2/home/liuzhonglan/wys/20250105/rift/include/geom/lithosphere_rift.h>
#include <aspect/geometry_model/box.h>
#include <aspect/utilities.h>

#include <cmath>

namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    void
    LithosphereRift<dim>::
    initialize ()
    {
      // Check that the required initial composition model is used
      const std::vector<std::string> active_initial_temperature_models = this->get_initial_temperature_manager().get_active_initial_temperature_names();
      AssertThrow(find(active_initial_temperature_models.begin(),active_initial_temperature_models.end(), "lithosphere with rift") != active_initial_temperature_models.end(),
                  ExcMessage("The lithosphere with rift initial composition plugin requires the lithosphere with rift initial temperature plugin."));
    }

    template <int dim>
    double
    LithosphereRift<dim>::
    initial_composition (const Point<dim> &position,
                         const unsigned int compositional_index) const
    {
      // Retrieve the indices of the fields that represent the lithospheric layers.
      // We assume a 3-layer system with an upper crust, lower crust, lithospheric mantle and heterogeneous mantle.
      const unsigned int id_upper = this->introspection().compositional_index_for_name("upper");
      const unsigned int id_lower = this->introspection().compositional_index_for_name("lower");
      const unsigned int id_mantle_upper = this->introspection().compositional_index_for_name("mantle_upper");
	  const unsigned int id_mantle_middle = this->introspection().compositional_index_for_name("mantle_middle");
	  const unsigned int id_mantle_lower = this->introspection().compositional_index_for_name("mantle_lower");
      const unsigned int id_grainsize = this->introspection().compositional_index_for_name("rift_grain_size");
	  const unsigned int id_water = this->introspection().compositional_index_for_name("water_content");
	  //20241227
	  const unsigned int id_max_melt = this->introspection().compositional_index_for_name("max_melt");

      // Determine coordinate system
      const bool cartesian_geometry = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()) != NULL ? true : false;

      const Point<dim-1> surface_point = surface_position(position, cartesian_geometry);

      // Get the distance to the line segments along a path parallel to the surface
      const double distance_to_rift_axis = distance_to_rift(surface_point);
	  
	  //20240702
	  //启用克拉通地幔时，克拉通和普通岩石圈使用两套计算模式
	  double local_upper_crust_thickness = 0;
	  double local_lower_crust_thickness = 0;
	  double local_mantle_upper_thickness = 0;
	  double local_mantle_middle_thickness = 0;
	  double local_mantle_lower_thickness = 0;
	  //20240704
	  //新增选项：边界内为克拉通，或边界外为克拉通
	  const bool center_is_cratontic = add_cratontic && center_cratontic && surface_point[0] >= cratontic_boundary[0] && surface_point[0] < cratontic_boundary[1];
	  const bool side_is_cratontic = add_cratontic && !center_cratontic &&(surface_point[0] < cratontic_boundary[0] || surface_point[0] >= cratontic_boundary[1]);
	  if (center_is_cratontic || side_is_cratontic)
	  {
		  local_upper_crust_thickness = cratontic_layer_thicknesses[0];
		  local_lower_crust_thickness = cratontic_layer_thicknesses[1];
		  local_mantle_upper_thickness = cratontic_layer_thicknesses[2];
		  local_mantle_middle_thickness = cratontic_layer_thicknesses[3];
		  local_mantle_lower_thickness = cratontic_layer_thicknesses[4];
	  }
	  else
	  {
		  // Get the signed distance to a polygon of different lithospheric thicknesses
          const std::pair<double, unsigned int> distance_to_L_polygon = distance_to_polygon(surface_point);

          // Compute the local thickness of the upper crust, lower crust and mantle part of the lithosphere
          // (in this exact order) based on the distance from the rift axis.
          local_upper_crust_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*polygon_thicknesses[distance_to_L_polygon.second][0]
                                                      +(0.5-0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*thicknesses[0])
                                                     * (!blend_rift_and_polygon && distance_to_L_polygon.first > 0.-2.*sigma_polygon ? 1. : (1.0 - A[0] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma_rift,2))))));
          local_lower_crust_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*polygon_thicknesses[distance_to_L_polygon.second][1]
                                                      +(0.5-0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*thicknesses[1])
                                                     * (!blend_rift_and_polygon && distance_to_L_polygon.first > 0.-2.*sigma_polygon ? 1. : (1.0 - A[1] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma_rift,2))))));
          local_mantle_upper_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*polygon_thicknesses[distance_to_L_polygon.second][2]
                                                             +(0.5-0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*thicknesses[2])
                                                            *  (!blend_rift_and_polygon && distance_to_L_polygon.first > 0.-2.*sigma_polygon ? 1. : (1.0 - A[2] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma_rift,2))))));
	      local_mantle_middle_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*polygon_thicknesses[distance_to_L_polygon.second][3]
                                                             +(0.5-0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*thicknesses[3])
                                                            *  (!blend_rift_and_polygon && distance_to_L_polygon.first > 0.-2.*sigma_polygon ? 1. : (1.0 - A[3] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma_rift,2))))));
	      local_mantle_lower_thickness = ((0.5+0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*polygon_thicknesses[distance_to_L_polygon.second][4]
                                                             +(0.5-0.5*std::tanh(distance_to_L_polygon.first/sigma_polygon))*thicknesses[4])
                                                            *  (!blend_rift_and_polygon && distance_to_L_polygon.first > 0.-2.*sigma_polygon ? 1. : (1.0 - A[4] * std::exp((-std::pow(distance_to_rift_axis,2)/(2.0*std::pow(sigma_rift,2))))));
	  }
	  
	  
      														
	  const double upper_crust = local_upper_crust_thickness;
	  const double lower_crust = local_upper_crust_thickness + local_lower_crust_thickness;
	  const double mantle_upper = local_upper_crust_thickness + local_lower_crust_thickness + local_mantle_upper_thickness;
	  const double mantle_middle = local_upper_crust_thickness + local_lower_crust_thickness + local_mantle_upper_thickness + local_mantle_middle_thickness;
	  const double mantle_lower = local_upper_crust_thickness + local_lower_crust_thickness + local_mantle_upper_thickness + local_mantle_middle_thickness + local_mantle_lower_thickness;

      // Compute depth
      const double depth = this->get_geometry_model().depth(position);

      // Check which layer we're in and return value.
      if (depth <= upper_crust && compositional_index == id_upper)
          return 1.;
      else if (depth > upper_crust && depth <= lower_crust && compositional_index == id_lower)
          return 1.;
	  //20240626
      else if (depth > lower_crust && depth <= mantle_upper && compositional_index == id_mantle_upper)
          return 1.;
	  else if (depth > mantle_upper && depth <= mantle_middle && compositional_index == id_mantle_middle)
		  return 1.;
	  else if (depth > mantle_middle && depth <= mantle_lower && compositional_index == id_mantle_lower)
		  return 1.;
	  // Check which layer we're in and return initial grain size
	  //upper
      else if (compositional_index == id_grainsize && depth <= upper_crust)
          return init_grain_size[id_upper + 1];
	  //lower
	  else if (compositional_index == id_grainsize && depth > upper_crust && depth <= lower_crust)
		  return init_grain_size[id_lower + 1];
	  //mantle
      //岩石圈地幔初始粒度规则：从莫霍面至LAB面，令10指数的绝对值按深度递增，斜率k = (LAB面粒度指数值 - 莫霍面粒度指数值) / 岩石圈地幔厚度
	  //设置粒度初始值时认定莫霍面为平面，如有地壳增厚的位置，岩石圈地幔的粒度按照距离初始莫霍面的位置设定。
	  //20240701
	  //计算岩石圈地幔粒度时，将所有岩石圈地幔放在一起
      else if (!exclude_mantle_lower && compositional_index == id_grainsize && depth > lower_crust && depth <= mantle_lower)
	  {
		  const double lg_grainL_prm = log10(init_grain_size[id_mantle_upper + 1]);
	      const double lg_grainA_prm = log10(init_grain_size[0]);
		  double lg_grain_k = 0;
		  double lg_grainL_init = 0;
		  
		  if(center_is_cratontic || side_is_cratontic)
		  {
			  lg_grain_k = (lg_grainA_prm - lg_grainL_prm) / (cratontic_layer_thicknesses[2] + cratontic_layer_thicknesses[3] + cratontic_layer_thicknesses[4]);
		      lg_grainL_init = lg_grainL_prm + lg_grain_k * (depth  - (cratontic_layer_thicknesses[0] + cratontic_layer_thicknesses[1]));
		  }
		  else
		  {
			  lg_grain_k = (lg_grainA_prm - lg_grainL_prm) / (thicknesses[2] + thicknesses[3] + thicknesses[4]);
		      lg_grainL_init = lg_grainL_prm + lg_grain_k * (depth  - (thicknesses[0] + thicknesses[1]));
		  }
		  return pow(10, lg_grainL_init);
	  }
	  else if (exclude_mantle_lower && compositional_index == id_grainsize && depth > lower_crust && depth <= mantle_middle)
	  {
		  const double lg_grainL_prm_ex = log10(init_grain_size[id_mantle_upper + 1]);
	      const double lg_grainA_prm_ex = log10(init_grain_size[0]);
		  double lg_grain_k_ex = 0;
		  double lg_grainL_init_ex = 0;
		  
		  if(center_is_cratontic || side_is_cratontic)
		  {
			  lg_grain_k_ex = (lg_grainA_prm_ex - lg_grainL_prm_ex) / (cratontic_layer_thicknesses[2] + cratontic_layer_thicknesses[3]);
		      lg_grainL_init_ex = lg_grainL_prm_ex + lg_grain_k_ex * (depth  - (cratontic_layer_thicknesses[0] + cratontic_layer_thicknesses[1]));
		  }
		  else
		  {
			  lg_grain_k_ex = (lg_grainA_prm_ex - lg_grainL_prm_ex) / (thicknesses[2] + thicknesses[3]);
		      lg_grainL_init_ex = lg_grainL_prm_ex + lg_grain_k_ex * (depth  - (thicknesses[0] + thicknesses[1]));
		  }
		  return pow(10, lg_grainL_init_ex);
	  }
	  else if (exclude_mantle_lower && compositional_index == id_grainsize && depth > mantle_middle && depth <= mantle_lower)
		  return init_grain_size[id_mantle_lower + 1];
	  //background 
	  else if (compositional_index == id_grainsize && depth > mantle_lower)
		  return init_grain_size[0];
	  //water content
	  //upper
	  else if (compositional_index == id_water && depth <= upper_crust)
		  return init_water_content[id_upper + 1];
	  //lower
	  else if (compositional_index == id_water && depth > upper_crust && depth <= lower_crust)
		  return init_water_content[id_lower + 1];
	  //20240626
	  //mantle_upper
	  else if (compositional_index == id_water && depth > lower_crust && depth <= mantle_upper)
		  return init_water_content[id_mantle_upper + 1];
	  //mantle_middle
	  else if (compositional_index == id_water && depth > mantle_upper && depth <= mantle_middle)
		  return init_water_content[id_mantle_middle + 1];
	  //mantle_lower
	  else if (compositional_index == id_water && depth > mantle_middle && depth <= mantle_lower)
		  return init_water_content[id_mantle_lower + 1];
	  //background
	  else if (compositional_index == id_water && depth > mantle_lower)
		  return init_water_content[0];
	  //20241227 给mantle_lower填入最大熔融值
	  //20241231 现在此功能适配于所有情况
	  else if (depth > mantle_middle && depth <= mantle_lower && compositional_index == id_max_melt && init_density_change_by_melt)
	  {
		  //斜率按照最厚的“mantle_lower”相设置
		  double max_polygon_thickness = 0.;
		  for (unsigned int a = 0; a < polygon_thicknesses.size(); a++)
			  max_polygon_thickness = std::max(polygon_thicknesses[a][4], max_polygon_thickness);
		  const double max_mantle_lower_thickness = std::max(thicknesses[4], std::max(max_polygon_thickness, cratontic_layer_thicknesses[4]));
		  AssertThrow(max_mantle_lower_thickness > 0., ExcMessage ("The thickness of 'mantle_lower' phase should be above 0 if the initial density modification is on."));
		  const double init_melt_k = init_max_melt / max_mantle_lower_thickness;
		  const double depth_in_mantle_lower = depth - mantle_middle;
		  const double output_init_melt = mantle_lower - mantle_middle > 1. ? init_max_melt - init_melt_k * depth_in_mantle_lower : 0.;
		  return output_init_melt;
	  }
	  else
          return 0.;
    }

    template <int dim>
    double
    LithosphereRift<dim>::
    distance_to_rift (const Point<dim-1> &surface_position) const
    {
      // Initiate distance with large value
      double distance_to_rift_axis = 1e23;
      double temp_distance = 0;

      // Loop over all line segments
      for (unsigned int i_segments = 0; i_segments < point_list.size(); ++i_segments)
        {
          temp_distance = (dim == 2) ? std::abs(surface_position[0]-point_list[i_segments][0][0]) : std::abs(Utilities::distance_to_line(point_list[i_segments], Point<2>(surface_position[0],surface_position[dim-2])));

          // Get the minimum distance
          distance_to_rift_axis = std::min(distance_to_rift_axis, temp_distance);
        }

      return distance_to_rift_axis;
    }

    template <int dim>
    Point<dim-1>
    LithosphereRift<dim>::
    surface_position (const Point<dim> &position,
                      const bool cartesian_geometry) const
    {
      Point<dim-1> surface_point;
      if (cartesian_geometry)
        {
          for (unsigned int d=0; d<dim-1; ++d)
            surface_point[d]=position[d];
        }
      // chunk (spherical) geometries
      else
        {
          // spherical coordinates in radius [m], lon [rad], colat [rad] format
          const std::array<double,dim> spherical_point = Utilities::Coordinates::cartesian_to_spherical_coordinates(position);
          // return lon [degrees], lat [degrees]
          for (unsigned int d=0; d<dim-1; ++d)
            surface_point[d] = spherical_point[d+1]*180./numbers::PI;
          if (dim == 3)
            surface_point[dim-2] = 90. - surface_point[dim-2];
        }

      return surface_point;
    }

    template <int dim>
    std::pair<double,unsigned int>
    LithosphereRift<dim>::
    distance_to_polygon (const Point<dim-1> &surface_position) const
    {
      // Inside is positive, outside negative. We assume no overlap of the different polygons.
      double max_distance = -1e24;
      unsigned int max_distance_polygon = 0;
      for (unsigned int n = 0; n<polygon_point_list.size(); ++n)
        {
          double temp_distance = 0;
          if (dim == 2)
            {
              double sign = -1.;
              if (surface_position[0]>polygon_point_list[n][0][0] && surface_position[0]<polygon_point_list[n][1][0])
                sign = 1.;
              temp_distance = sign * std::min(std::abs(polygon_point_list[n][1][0] - surface_position[0]), std::abs(surface_position[0] - polygon_point_list[n][0][0]));
            }
          else
            {
              temp_distance = Utilities::signed_distance_to_polygon<dim>(polygon_point_list[n], Point<2>(surface_position[0],surface_position[dim-2]));
            }

          if (temp_distance > max_distance)
            {
              max_distance = temp_distance;
              max_distance_polygon = n;
            }
        }
      return std::pair<double, unsigned int> (max_distance, max_distance_polygon);
    }

    template <int dim>
    void
    LithosphereRift<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Initial composition model");
      {
        prm.enter_subsection("Lithosphere with rift");
        {
          prm.declare_entry ("Standard deviation of Gaussian rift geometry", "20000",
                             Patterns::Double (0),
                             "The standard deviation of the Gaussian distribution of the amplitude of the strain noise. "
                             "Note that this parameter is taken to be the same for all rift segments. "
                             "Units: $m$ or degrees.");
          prm.declare_entry ("Half width of polygon smoothing", "20000",
                             Patterns::Double (0),
                             "The half width of the hyperbolic tangent smoothing used to transition to the lithospheric thicknesses of the polygon. "
                             "Note that this parameter is taken to be the same for all polygons. "
                             "Units: $m$ or degrees.");
          prm.declare_entry ("Blend polygons and rifts", "true",
                             Patterns::Bool (),
                             "Whether or not to blend the contributions of polygons and the rift. For true, they're blend together. "
                             "For false, the polygon thicknesses are taken as the local thicknesses. "
                             "Units: /.");
          prm.declare_entry ("Amplitude of Gaussian rift geometry", "0.2",
                             Patterns::List(Patterns::Double (-1,1)),
                             "The amplitude of the Gaussian distribution of the amplitude of the strain noise. "
                             "Note that this parameter is taken to be the same for all rift segments, but can "
                             "vary per lithosphere layer. "
                             "Units: none.");
          prm.declare_entry ("Layer thicknesses", "30000.",
                             Patterns::List(Patterns::Double(0)),
                             "List of thicknesses for the bottom of the lithospheric layers,"
                             "for a total of N+1 values, where N is the number of compositional fields."
                             "If only one value is given, then all use the same value.  Units: $m$");
          prm.declare_entry ("Rift axis line segments",
                             "",
                             Patterns::Anything(),
                             "Set the line segments that represent the rift axis. Each segment is made up of "
                             "two points that represent horizontal coordinates (x,y) or (lon,lat). "
                             "The exact format for the point list describing the segments is "
                             "\"x1,y1>x2,y2;x2,y2>x3,y3;x4,y4>x5,y5\". Note that the segments can be connected "
                             "or isolated. The units of the coordinates are "
                             "dependent on the geometry model. In the box model they are in meters, in the "
                             "chunks they are in degrees.");
          prm.declare_entry ("Lithospheric polygons",
                             "",
                             Patterns::List(Patterns::Anything()),
                             "Set the polygons that represent an area of different lithospheric thickness. "
                             "The polygons are separated by semicolons."
                             "Each polygon is a list of "
                             "points that represent horizontal coordinates (x,y) or (lon,lat). "
                             "The exact format for the point list describing a polygon is "
                             "\"x1,y1>x2,y2>x3,y3>x4,y4>x5,y5\". Note that the polygon is assumed to be closed. "
                             "The units of the coordinates are "
                             "dependent on the geometry model. In the box model they are in meters, in the "
                             "chunks they are in degrees.");
          prm.declare_entry ("Lithospheric polygon layer thicknesses", "30000.",
                             Patterns::List(Patterns::List(Patterns::Double(0),0,5,","),0,10,";"),
                             "List of thicknesses of the lithospheric layers for each polygon."
                             "For each polygon a total of 5 values should be given (upper crust, lower crust, mantle lithosphere, heterogeneous mantle)."
                             "If only one value is given, then all use the same value.  Units: $m$");
							 
		  prm.declare_entry ("Add cratontic lithosphere", "false",
                             Patterns::Bool (),
                             "Whether or not to add Cratontic Lithosphere in the model. For true, a thicker lithosphere will be set in the right side. "
							 "Note that this function is only available in 2D models."
                             "Units: /.");
		  prm.declare_entry ("Segment of cratontic lithosphere boundary", "20000",
                             Patterns::List(Patterns::Double(0)),
                             "Segment of cratontic lithosphere boundary. "
                             "Units: $m$.");		 
		  prm.declare_entry ("Cratontic lithosphere layer thicknesses", "30000.",
                           Patterns::List(Patterns::Double(0)),
                           "List of thicknesses for the bottom of the cratontic lithospheric layers,"
                           "for a total of N+1 values, where N is the number of compositional fields."
                           "If only one value is given, then all use the same value.  Units: $m$");
		  //20240704
		  //新增选项：克拉通在中心还是在两侧
		  prm.declare_entry ("Inside the boundaries is cratontic lithosphere", "true",
                             Patterns::Bool (),
                             "Whether or not to set Cratontic Lithosphere to the center "
							 "Note that this function is only available in 2D models."
                             "Units: /.");
							 
		  //20240716
		  //新增选项：mantle_lower是否算作地幔物质
		  prm.declare_entry ("Exclude phase 'mantle_lower' from mantle material", "false",
                             Patterns::Bool (),
                             "Whether or not to exclude phase 'mantle_lower' from mantle material." 
							 "For true, phase 'mantle_lower' will not be included in initial temperature model. "
							 "Note that this function is only available in 2D models."
                             "Units: /.");
							 
	      //20241227
		  //新增：使用最大熔融程度对密度进行修正
		  prm.declare_entry ("Use 'melt_max' to modify the initial density of 'mantle_lower'", "false",
                             Patterns::Bool (),
                             "Use 'melt_max' to modify the density of 'mantle_lower'." 
							 "For true, phase 'mantle_lower' will be set a initial 'max_melt' to modify the density. "
							 "Note that this function is only available in 2D models."
                             "Units: /.");
		  prm.declare_entry ("Max melt fraction in initial density modification", "0.",
                             Patterns::Double (0),
                             "Max melt fraction in initial density modification." 
							 "The initial max melt fraction will be changed linear via depth in the 'mantle_lower' phase. "
                             "Units: /.");					 

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    LithosphereRift<dim>::parse_parameters (ParameterHandler &prm)
    {
      // Check that the required compositional fields exist.
      AssertThrow(this->introspection().compositional_name_exists("upper"),ExcMessage("We need a compositional field called 'upper' representing the upper crust."));
      AssertThrow(this->introspection().compositional_name_exists("lower"),ExcMessage("We need a compositional field called 'lower' representing the lower crust."));
      AssertThrow(this->introspection().compositional_name_exists("mantle_upper"),ExcMessage("We need a compositional field called 'mantle_upper' representing the lithospheric part of the mantle."));
	  AssertThrow(this->introspection().compositional_name_exists("mantle_middle"),ExcMessage("We need a compositional field called 'mantle_middle' representing the heterogeneous part of the mantle."));
	  AssertThrow(this->introspection().compositional_name_exists("mantle_lower"),ExcMessage("We need a compositional field called 'mantle_lower' representing the heterogeneous part of the mantle."));
      AssertThrow(this->introspection().compositional_name_exists("rift_grain_size"), ExcMessage("We need a compositional field called 'rift_grain_size' representing the grain size of every lithospheric part."));

      prm.enter_subsection ("Initial composition model");
      {
        prm.enter_subsection("Lithosphere with rift");
        {
          sigma_rift             = prm.get_double ("Standard deviation of Gaussian rift geometry");
          sigma_polygon          = prm.get_double ("Half width of polygon smoothing");
          blend_rift_and_polygon = prm.get_bool ("Blend polygons and rifts");
          A                      = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Amplitude of Gaussian rift geometry"))),
                                                                           5,
                                                                           "Amplitude of Gaussian rift geometry");
          thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Layer thicknesses"))),
                                                                5,
                                                                "Layer thicknesses");
          // Read in the string of segments
          const std::string temp_all_segments = prm.get("Rift axis line segments");
          // Split the string into segment strings
          const std::vector<std::string> temp_segments = Utilities::split_string_list(temp_all_segments,';');
          const unsigned int n_temp_segments = temp_segments.size();
          point_list.resize(n_temp_segments);
          // Loop over the segments to extract the points
          for (unsigned int i_segment = 0; i_segment < n_temp_segments; i_segment++)
            {
              // In 3d a line segment consists of 2 points,
              // in 2d only 1 (ridge axis orthogonal to x and y)

              const std::vector<std::string> temp_segment = Utilities::split_string_list(temp_segments[i_segment],'>');

              if (dim == 3)
                {
                  AssertThrow(temp_segment.size() == 2,ExcMessage ("The given coordinate '" + temp_segment[i_segment] + "' is not correct. "
                                                                   "It should only contain 2 parts: "
                                                                   "the two points of the segment, separated by a '>'."));
                }
              else
                {
                  AssertThrow(temp_segment.size() == 1,ExcMessage ("The given coordinate '" + temp_segment[i_segment] + "' is not correct. "
                                                                   "It should only contain 1 part: "
                                                                   "the point representing the rift axis."));
                }

              // Loop over the dim-1 points of each segment (i.e. in 2d only 1 point is required for a 'segment')
              for (unsigned int i_points = 0; i_points < dim-1; i_points++)
                {
                  const std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_segment[i_points],','));
                  if (dim == 3)
                    {
                      AssertThrow(temp_point.size() == 2,ExcMessage ("The given coordinates of segment '" + temp_segment[i_points] + "' are not correct. "
                                                                "It should only contain 2 parts: "
                                                                "the two coordinates of the segment end point, separated by a ','."));
                    }
                  else
                    {
                      AssertThrow(temp_point.size() == 1,ExcMessage ("The given coordinates of segment '" + temp_segment[i_points] + "' are not correct. "
                                                                "It should only contain 1 part: "
                                                                "the one coordinate of the segment end point."));
                    }

                      // Add the point to the list of points for this segment
                      point_list[i_segment][i_points][0] = temp_point[0];
                      point_list[i_segment][i_points][1] = temp_point[dim-2];
                }
            }

          // Split the string into the separate polygons
          const std::vector<std::string> temp_polygons = Utilities::split_string_list(prm.get("Lithospheric polygons"),';');
          const std::vector<std::string> temp_thicknesses = Utilities::split_string_list(prm.get("Lithospheric polygon layer thicknesses"),';');
          const unsigned int n_polygons = temp_polygons.size();
          AssertThrow(temp_thicknesses.size() == n_polygons || temp_thicknesses.size() == 1,
                      ExcMessage("The number of polygons does not correspond to the number of polygons for which a thickness is prescribed."));
          polygon_point_list.resize(n_polygons);
          polygon_thicknesses.resize(n_polygons);
          for (unsigned int i_polygons = 0; i_polygons < n_polygons; ++i_polygons)
            {
              polygon_thicknesses[i_polygons] = Utilities::string_to_double(Utilities::split_string_list(temp_thicknesses[i_polygons],','));
              AssertThrow(polygon_thicknesses[i_polygons].size()==5, ExcMessage ("The number of layer thicknesses should be equal to 5."));

              // Split the string into point strings
              const std::vector<std::string> temp_points = Utilities::split_string_list(temp_polygons[i_polygons],'>');
              const unsigned int n_temp_points = temp_points.size();
              if (dim == 3)
                {
                  AssertThrow(n_temp_points>=3, ExcMessage ("The number of polygon points should be equal to or larger than 3 in 3d."));
                }
              else
                {
                  AssertThrow(n_temp_points==2, ExcMessage ("The number of polygon points should be equal to 2 in 2d."));
                }
              polygon_point_list[i_polygons].resize(n_temp_points);
              // Loop over the points of the polygon. Each point should consist of 2 values (lon and lat coordinate).
              for (unsigned int i_points = 0; i_points < n_temp_points; i_points++)
                {
                  const std::vector<double> temp_point = Utilities::string_to_double(Utilities::split_string_list(temp_points[i_points],','));
                  AssertThrow(temp_point.size() == dim-1,ExcMessage ("The given coordinates of point '" + temp_points[i_points] + "' are not correct. "
                                                                "It should only contain 1 (2d) or 2 (in 3d) parts: "
                                                                "the longitude/x (and latitude/y in 3d) coordinate (separated by a ',')."));

                  // Add the point to the list of points for this segment
                  polygon_point_list[i_polygons][i_points][0] = temp_point[0];
                  polygon_point_list[i_polygons][i_points][1] = temp_point[dim-2];
                }
              if  (dim == 2)
                AssertThrow(polygon_point_list[i_polygons][0][0] < polygon_point_list[i_polygons][1][0], ExcMessage("The order of the x coordinates of the 2 points "
                            "of each 2d polygon should be ascending. "));
							
			  add_cratontic = prm.get_bool ("Add cratontic lithosphere");
			  cratontic_boundary = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Segment of cratontic lithosphere boundary"))),
                                                                2,
                                                                "Segment of cratontic lithosphere boundary");
			  cratontic_layer_thicknesses = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Cratontic lithosphere layer thicknesses"))),
                                                                5,
                                                                "Cratontic lithosphere layer thicknesses");
			  center_cratontic = prm.get_bool ("Inside the boundaries is cratontic lithosphere");
			  //20240716
		      exclude_mantle_lower = prm.get_bool ("Exclude phase 'mantle_lower' from mantle material");
			  
			  //20241227
			  //使用最大熔融程度对密度进行修正
			  init_density_change_by_melt = prm.get_bool ("Use 'melt_max' to modify the initial density of 'mantle_lower'");
			  init_max_melt = prm.get_double ("Max melt fraction in initial density modification");
			  
			  

            }
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      unsigned int n_fields = 0;
      prm.enter_subsection("Compositional fields");
      {
          n_fields = prm.get_integer("Number of fields");
      }
      prm.leave_subsection();

      prm.enter_subsection("Material model");
      {
          prm.enter_subsection("Visco Plastic Grain Size");
          {
              init_grain_size = Utilities::possibly_extend_from_1_to_N(Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial Grain size"))),
                                n_fields + 1,
                                "Initial Grain size");
			  init_water_content = Utilities::possibly_extend_from_1_to_N(Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial water content"))),
                                n_fields + 1,
                                "Initial water content");
          }
          prm.leave_subsection();
      }
      prm.leave_subsection();
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialComposition
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(LithosphereRift,
                                              "lithosphere with rift",
                                              "A class that implements initial conditions for the porosity field "
                                              "by computing the equilibrium melt fraction for the given initial "
                                              "condition and reference pressure profile. Note that this plugin only "
                                              "works if there is a compositional field called `porosity', and the "
                                              "used material model implements the 'MeltFractionModel' interface. "
                                              "For all compositional fields except porosity this plugin returns 0.0, "
                                              "and they are therefore not changed as long as the default `add' "
                                              "operator is selected for this plugin.")
  }
}
