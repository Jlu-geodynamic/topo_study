/*
  Copyright (C) 2015 - 2022 by the authors of the ASPECT code.

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


#include </fs2/home/liuzhonglan/wys/20250105/grain/include/heat/latent_heat_viscoplastic_melt.h>
#include <aspect/material_model/interface.h>


namespace aspect
{
  namespace HeatingModel
  {
    template <int dim>
    void
    LatentHeatViscoPlasticMelt<dim>::
    evaluate (const MaterialModel::MaterialModelInputs<dim> &material_model_inputs,
              const MaterialModel::MaterialModelOutputs<dim> &material_model_outputs,
              HeatingModel::HeatingModelOutputs &heating_model_outputs) const
    {
      for (unsigned int q=0; q<heating_model_outputs.heating_source_terms.size(); ++q)
        {
		  heating_model_outputs.heating_source_terms[q] = 0.0;
		  heating_model_outputs.lhs_latent_heat_terms[q] = 0.0;
		  heating_model_outputs.rates_of_temperature_change[q] = 0.0;
		  if (this->introspection().compositional_name_exists("melt_fraction") &&  this->get_timestep_number() > 0)
		  {
			  const double last_melt_fractions = material_model_inputs.composition[q][this->introspection().compositional_index_for_name("melt_fraction")];
			  const double melt_fractions_change = material_model_outputs.reaction_terms[q][this->introspection().compositional_index_for_name("melt_fraction")];
			  //混合熔融时，根据成分判定当前位置的熔融类型
              const double target_mantle_fraction_change = include_mantle_melting && this->introspection().compositional_name_exists("new_crust") ? 
			                                               material_model_outputs.reaction_terms[q][this->introspection().compositional_index_for_name("new_crust")]
														 : 0.;
			  const double target_crust_fraction_change = include_crustal_melting && this->introspection().compositional_name_exists("core_complex") ? 
			                                               material_model_outputs.reaction_terms[q][this->introspection().compositional_index_for_name("core_complex")]
														 : 0.;
			  const double crust_fraction = include_crustal_melting ? 
			                                material_model_inputs.composition[q][this->introspection().compositional_index_for_name("upper")]
			                              + material_model_inputs.composition[q][this->introspection().compositional_index_for_name("lower")]
										  + material_model_inputs.composition[q][this->introspection().compositional_index_for_name("new_crust")]
										  : 0.;
			  const double mantle_fraction = include_mantle_melting ? 
			                               1. - material_model_inputs.composition[q][this->introspection().compositional_index_for_name("sediment_1")]
			                               - material_model_inputs.composition[q][this->introspection().compositional_index_for_name("sediment_2")]
										   - material_model_inputs.composition[q][this->introspection().compositional_index_for_name("upper")]
										   - material_model_inputs.composition[q][this->introspection().compositional_index_for_name("lower")]
										   - material_model_inputs.composition[q][this->introspection().compositional_index_for_name("mantle_upper")]
										   - target_mantle_fraction_change
										   - target_crust_fraction_change
										   : 0.;
			  
			  
			  //当时间步长过小时，熔融造成的降温会非常严重
			  //因此将最小的时间步长限制在2500yr
			  const double time = std::max(this->get_timestep(), 2500 * 3600 * 24 * 365.25);
			  
			  //1.熔融吸热导致的温度变化
			  //在程序中与放射热的形式相同
			  //等式右侧单位 
			  //mantle_melt_temp_reduction/crustal_melt_temp_reduction: J*kg-1*K-1
			  //melt_fractions: none
			  //temperature: K
			  //densities: kg*m-3
			  //time: s
				  
		      //等式左侧单位
		      //heating_source_terms：J*s-1*m-3 (Watt*m-3)，
		      //从物理意义的角度出发，heating_source_terms 反映辐射热对温度的影响（radioactive heat production，Hr）
			  //在模型当中，adiabatic heat、latent heat、radioactive heat production 三项为并列关系
			  //因此 adiabatic heat、latent heat 在计算过程中被转换为和 radioactive heat production 相同的单位。
			  
			  //2.洋壳放热导致的温度变化
			  //根据不同熔融机制形成的产物进行区分
			  
			  //判定熔融形式，混合型与独立型的计算规则有区别
			  if (include_mantle_melting && !include_crustal_melting)
				  heating_model_outputs.heating_source_terms[q] = last_melt_fractions > 1e-3 && melt_fractions_change > 1e-5 ?
			                                                      - mantle_melt_temp_reduction
			                                                      * melt_fractions_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time
																: 0.;
				  
			  else if (!include_mantle_melting && include_crustal_melting)
				  heating_model_outputs.heating_source_terms[q] = last_melt_fractions > 1e-3 && melt_fractions_change > 1e-5 ?
				                                                  - crustal_melt_temp_reduction
			                                                      * melt_fractions_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time
																: 0.;
																
			  else if (include_mantle_melting && include_crustal_melting)
			  {
				  if (mantle_fraction >= crust_fraction)
					  heating_model_outputs.heating_source_terms[q] = last_melt_fractions > 1e-3 && melt_fractions_change > 1e-5 ?
			                                                      - mantle_melt_temp_reduction
			                                                      * melt_fractions_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time
																: 0.;
				  else
					  heating_model_outputs.heating_source_terms[q] = last_melt_fractions > 1e-3 && melt_fractions_change > 1e-5 ?
				                                                  - crustal_melt_temp_reduction
			                                                      * melt_fractions_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time
																: 0.;
			  }
				  
			  else
				  heating_model_outputs.heating_source_terms[q] = 0.;
			  
              //2.洋壳放热导致的温度变化
			  //根据不同熔融机制形成的产物进行区分
			  //为了不与熔融吸热过程冲突，放热过程不使用ifelse，均为独立判定，仅在新生物质的成分满足条件时赋值

			  //仅有地幔熔融
			  if (!include_crustal_melting && target_mantle_fraction_change > 0.1)
				  heating_model_outputs.heating_source_terms[q] = mantle_melt_temp_reduction 
				                                                  * target_mantle_fraction_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time;
			  //仅有地壳熔融
			  if (!include_mantle_melting && target_crust_fraction_change > 0.1)
				  heating_model_outputs.heating_source_terms[q] = crustal_melt_temp_reduction 
				                                                  * target_crust_fraction_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time;
			  //混合熔融
			  //暂定取成分高的熔融信息
			  if (include_crustal_melting && include_mantle_melting && (target_mantle_fraction_change > 0.1 || target_crust_fraction_change > 0.1))
			  {
				  const double target_mantle_change_for_composite = target_mantle_fraction_change > 0.1 ? target_mantle_fraction_change : 0.;
				  const double target_crust_change_for_composite = target_crust_fraction_change > 0.1 ? target_crust_fraction_change : 0.;
				  heating_model_outputs.heating_source_terms[q] = target_mantle_change_for_composite >= target_crust_change_for_composite ?
				                                                  mantle_melt_temp_reduction 
				                                                  * target_mantle_fraction_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time
																: crustal_melt_temp_reduction 
				                                                  * target_crust_fraction_change
		                                                          * material_model_inputs.temperature[q] 
														          * material_model_outputs.densities[q]
														          / time;

			}
			  
		}
    }
  }

    template <int dim>
    void
    LatentHeatViscoPlasticMelt<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Latent heat viscoplastic melt");
        {
          prm.declare_entry ("Change of entropy on mantle melting", "400.",
                             Patterns::Double (),
                             "Change of entropy on mantle melting "
                             "Units: J*kg-1*K-1.");
		  prm.declare_entry ("Change of entropy on crustal melting", "250.",
                             Patterns::Double (),
                             "Change of entropy on crustal melting "
                             "Units: J*kg-1*K-1.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    LatentHeatViscoPlasticMelt<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Latent heat viscoplastic melt");
        {
          mantle_melt_temp_reduction = prm.get_double ("Change of entropy on mantle melting");
		  crustal_melt_temp_reduction = prm.get_double ("Change of entropy on crustal melting");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
	  
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
  namespace HeatingModel
  {
    ASPECT_REGISTER_HEATING_MODEL(LatentHeatViscoPlasticMelt,
                                  "latent heat viscoplastic melt",
                                  "Implementation of a standard model for latent heat "
                                  "of melting. This assumes that there is a compositional field "
                                  "called porosity, and it uses the reaction term of this field "
                                  "(the fraction of material that melted in the current time step) "
                                  "multiplied by a constant entropy change for melting all "
                                  "of the material as source term of the heating model.\n"
                                  "If there is no field called porosity, the heating terms are 0.")
  }
}
