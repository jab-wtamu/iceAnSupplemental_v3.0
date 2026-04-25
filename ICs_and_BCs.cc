#include "custom_pde.h"

#include <prismspf/core/initial_conditions.h>
#include <prismspf/core/nonuniform_dirichlet.h>

#include <prismspf/user_inputs/user_input_parameters.h>

#include <prismspf/config.h>

#include <cmath>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_initial_condition(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{
  const std::array<double, 3> center = {
    {0.5, 0.5, 0.5}
  };

  const double rad_xy = 15.0;
  const double rad_z  = 4.0;

  if (index == 0)
    {
      // Variable 0: u
      scalar_value = u0;
    }
  else if (index == 1)
    {
      const auto &sizes =
        this->get_user_inputs().get_spatial_discretization().get_size();

      const double xc = center[0] * sizes[0];
      const double yc = (dim > 1) ? center[1] * sizes[1] : 0.0;
      const double zc = (dim > 2) ? center[2] * sizes[2] : 0.0;

      const double dx = point[0] - xc;
      const double dy = (dim > 1) ? point[1] - yc : 0.0;
      const double dz = (dim > 2) ? point[2] - zc : 0.0;

      const double r_ell =
        std::sqrt((dx * dx + dy * dy) / (rad_xy * rad_xy) +
                  (dz * dz) / (rad_z * rad_z));

      scalar_value = -std::tanh((r_ell - 1.0) / std::sqrt(2.0));
    }
  else if (index == 2)
    {
      // Variable 2: xi1
      scalar_value = 0.0;
    }
}

template <unsigned int dim, unsigned int degree, typename number>
void
CustomPDE<dim, degree, number>::set_nonuniform_dirichlet(
  [[maybe_unused]] const unsigned int       &index,
  [[maybe_unused]] const unsigned int       &boundary_id,
  [[maybe_unused]] const unsigned int       &component,
  [[maybe_unused]] const dealii::Point<dim> &point,
  [[maybe_unused]] number                   &scalar_value,
  [[maybe_unused]] number                   &vector_component_value) const
{}

#include "custom_pde.inst"

PRISMS_PF_END_NAMESPACE
