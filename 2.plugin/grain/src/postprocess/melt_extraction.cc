/*
  Copyright (C) 2011 - 2019 by the authors of the ASPECT code.

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


#include </fs2/home/liuzhonglan/wys/20250105/grain/include/postprocess/melt_extraction.h>
#include </fs2/home/liuzhonglan/wys/20250105/grain/include/postprocess/pre_melt_extraction.h>

#include <aspect/geometry_model/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <dirent.h>
#include <sys/stat.h>

//#include <mpi.h>

namespace aspect
{
  namespace Postprocess
  {
	  
	  
	  
	template <int dim>
	double
	MeltExtraction<dim>::
	read_from_txt(const std::string path,
	              const unsigned int mpi_communicator)const
	{
		double temp_input = 0;
		if(this->simulator_is_past_initialization() && this->get_timestep_number() > 0 && Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == mpi_communicator)
		{
			std::ifstream data_from_txt(path.c_str());
			std::string line;
			getline(data_from_txt, line);
			while(line == "")
			{
				data_from_txt.close();
				sleep(1.);
				std::ifstream data_from_txt(path.c_str());
				getline(data_from_txt, line);
			}
			std::stringstream ss(line);
			ss >> temp_input;
			data_from_txt.close();
			return temp_input;
		} 
		return temp_input;
	}	

    template <int dim>
    std::pair<std::string,std::string>
    MeltExtraction<dim>::execute (TableHandler &statistics)
    {
      if (this->n_compositional_fields() == 0)
        return std::pair<std::string,std::string>();

      // create a quadrature formula based on the compositional element alone.
      // be defensive about determining that a compositional field actually exists
      AssertThrow (this->introspection().base_elements.compositional_fields
                   != numbers::invalid_unsigned_int,
                   ExcMessage("This postprocessor cannot be used without compositional fields."));
				   
	  
      const QGauss<dim> quadrature_formula (this->get_fe().base_element(this->introspection().base_elements.compositional_fields).degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature_formula,
                               update_values   |
                               update_quadrature_points |
                               update_JxW_values);

	  MaterialModel::MaterialModelInputs<dim> in(fe_values.n_quadrature_points, this->n_compositional_fields());
	  unsigned int mantle_local_min_x_crood = 9999;
	  unsigned int mantle_local_max_x_crood = 0;
	  unsigned int crustal_local_min_x_crood = 9999;
	  unsigned int crustal_local_max_x_crood = 0;

      // compute the integral quantities by quadrature
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
			//获取mpi通道编号
			unsigned int mpi_comm_id = 0;
			unsigned int out_comm_id = 0;
		    while (out_comm_id < 500)
		    {
			    if (Utilities::MPI::this_mpi_process (this->get_mpi_communicator()) == out_comm_id)
			    {
				    mpi_comm_id = out_comm_id;
				    break;
			    } 
			    out_comm_id ++;
		    } 
			//填充输入
            fe_values.reinit (cell);
			in.reinit(fe_values, cell, this->introspection(), this->get_solution());
			const unsigned int mid_point = n_q_points / 2;
			const double posx = in.position[mid_point][0];
	        const unsigned int crood_x = std::floor( posx / ((max_x / repetitions_x) / std::pow(2, melt_fraction_refinement)));
			//20240703
			if(!vertical_trans)
			{
				if(include_mantle_melting)
				{
					const double mantle_emplace_location_y = read_from_txt(melt_info_path + "mantle_emplace_location_y.txt", mpi_comm_id);
					for (unsigned int q=0; q<n_q_points; ++q)
					{
						const double posy = in.position[q][1];
						  if (posy / 1e3 >= mantle_emplace_location_y - 6 && posy / 1e3 < mantle_emplace_location_y  && in.temperature[q] >= 1473)
						{
							mantle_local_min_x_crood = crood_x;
							mantle_local_max_x_crood = crood_x;
						}
					}
				}
				if(include_crustal_melting)
				{
					const double crustal_emplace_location_y = read_from_txt(melt_info_path + "crust_emplace_location_y.txt", mpi_comm_id);
					for (unsigned int q=0; q<n_q_points; ++q)
					{
						const double posy = in.position[q][1];
						  if (posy / 1e3 >= crustal_emplace_location_y - 6 && posy / 1e3 < crustal_emplace_location_y  && in.temperature[q] >= 873)
						{
							crustal_local_min_x_crood = crood_x;
							crustal_local_max_x_crood = crood_x;
						}
					}
				}
			}
			else if(vertical_trans)
			{
				mantle_local_min_x_crood = 0;
				mantle_local_max_x_crood = 0;
				crustal_local_min_x_crood = 0.;
				crustal_local_max_x_crood = 0.;
			}
			
          }
		  
	  const unsigned int global_mantle_min_x_crood = Utilities::MPI::min (mantle_local_min_x_crood, this->get_mpi_communicator());
	  const unsigned int global_mantle_max_x_crood = Utilities::MPI::max (mantle_local_max_x_crood, this->get_mpi_communicator());
	  
	  const unsigned int global_crustal_min_x_crood = Utilities::MPI::min (crustal_local_min_x_crood, this->get_mpi_communicator());
	  const unsigned int global_crustal_max_x_crood = Utilities::MPI::max (crustal_local_max_x_crood, this->get_mpi_communicator());
	  
	  //根据具体情况将熔体传送至地壳增生窗口，模型最大熔融程度小于0.2%时无法开启传输，期间产生的熔体会流失
	  //这部分熔体量非常少，对地壳增生的影响可以忽略不计。
	  
	  //当最大值大于0时，代表存在地壳增生窗口
	  const double min_mantle_emplace_window = global_mantle_max_x_crood > 0 ? global_mantle_min_x_crood : 0;
	  const double max_mantle_emplace_window = global_mantle_max_x_crood > 0 ? global_mantle_max_x_crood : 0;
	  const double min_mantle_emplace_window_in_km = min_mantle_emplace_window * ((max_x / repetitions_x) / std::pow(2, melt_fraction_refinement)) / 1e3;
	  const double max_mantle_emplace_window_in_km = max_mantle_emplace_window * ((max_x / repetitions_x) / std::pow(2, melt_fraction_refinement)) / 1e3;
	  
	  const double min_crustal_emplace_window = global_crustal_max_x_crood > 0 ? global_crustal_min_x_crood : 0;
	  const double max_crustal_emplace_window = global_crustal_max_x_crood > 0 ? global_crustal_max_x_crood : 0;
	  const double min_crustal_emplace_window_in_km = min_crustal_emplace_window * ((max_x / repetitions_x) / std::pow(2, melt_fraction_refinement)) / 1e3;
	  const double max_crustal_emplace_window_in_km = max_crustal_emplace_window * ((max_x / repetitions_x) / std::pow(2, melt_fraction_refinement)) / 1e3;
	  
	  if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
	  {
		  if(include_mantle_melting)
		  {
			  std::ofstream mantle_write_min_win((melt_info_path + "mantle_min_emplace_window.txt").c_str());
			  if(mantle_write_min_win)
				  mantle_write_min_win << min_mantle_emplace_window;
			  mantle_write_min_win.close();
			  
			  std::ofstream mantle_write_max_win((melt_info_path + "mantle_max_emplace_window.txt").c_str());
			  if(mantle_write_max_win)
				  mantle_write_max_win << max_mantle_emplace_window;
			  mantle_write_max_win.close();
		  }
		  
		  if(include_crustal_melting)
		  {
			  std::ofstream crustal_write_min_win((melt_info_path + "crust_min_emplace_window.txt").c_str());
			  if(crustal_write_min_win)
				  crustal_write_min_win << min_crustal_emplace_window;
			  crustal_write_min_win.close();
			  
			  std::ofstream crustal_write_max_win((melt_info_path + "crust_max_emplace_window.txt").c_str());
			  if(crustal_write_max_win)
				  crustal_write_max_win << max_crustal_emplace_window;
			  crustal_write_max_win.close();
		  }
		  
	  }
	  
      statistics.add_value ("Min emplacing window(mantle,km)", min_mantle_emplace_window_in_km);
	  statistics.add_value ("Max emplacing window(mantle,km)", max_mantle_emplace_window_in_km);
	  statistics.add_value ("Min emplacing window(crustal,km)", min_crustal_emplace_window_in_km);
	  statistics.add_value ("Max emplacing window(crustal,km)", max_crustal_emplace_window_in_km);
	  {
        const char *columns[] = {"Min emplacing window(mantle,km)", "Max emplacing window(mantle,km)", "Min emplacing window(crustal,km)", "Max emplacing window(crustal,km)"};
        for (unsigned int i=0; i<sizeof(columns)/sizeof(columns[0]); ++i)
          {
            statistics.set_precision (columns[i], 8);
            statistics.set_scientific (columns[i], true);
          }
      }
      std::ostringstream output;
      output.precision(4);
	  output << min_mantle_emplace_window_in_km << "km to "
	         << max_mantle_emplace_window_in_km << "km /"
			 << min_crustal_emplace_window_in_km << "km to"
			 << max_crustal_emplace_window_in_km << "km";
      return std::pair<std::string, std::string> ("Extracting melt, emplacing window mantle/crustal:",
                                                  output.str());
    }
	
	template <int dim>
    void
    MeltExtraction<dim>::parse_parameters (ParameterHandler &prm)
	{
		//20240703
		 prm.enter_subsection("Postprocess");
		{
			prm.enter_subsection("Melt extraction");
			{
				vertical_trans = prm.get_bool("Vertical transportation mode");
			}
			prm.leave_subsection ();
		}
		prm.leave_subsection (); 
		
		
		//20240614
		prm.enter_subsection("Mesh refinement");
		{
			melt_fraction_refinement = prm.get_integer("Initial adaptive refinement") + prm.get_integer("Initial global refinement");
			
		}
		prm.leave_subsection ();
		
		prm.enter_subsection("Geometry model");
		{
			prm.enter_subsection("Box");
			{
				max_x = prm.get_double ("X extent");
				repetitions_x = prm.get_integer ("X repetitions");
				melt_info_path = this->get_output_directory() + "melt_info/";
			}
			prm.leave_subsection ();
		}
		prm.leave_subsection ();
		
		prm.enter_subsection("Material model");
		{
			prm.enter_subsection ("Visco Plastic Grain Size");
			{
				include_mantle_melting = prm.get_bool ("Include mantle melting");
				include_crustal_melting = prm.get_bool("Include crustal melting");
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
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(MeltExtraction,
                                  "melt extraction",
                                  "A postprocessor that computes some statistics about "
                                  "the compositional fields, if present in this simulation. "
                                  "In particular, it computes maximal and minimal values of "
                                  "each field, as well as the total mass contained in this "
                                  "field as defined by the integral "
                                  "$m_i(t) = \\int_\\Omega c_i(\\mathbf x,t) \\; \\text{d}x$.")
  }
}
