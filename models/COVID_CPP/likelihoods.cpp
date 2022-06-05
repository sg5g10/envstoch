#include <stdio.h>      
#include <math.h>       
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

#define PI 3.14159265

namespace py = pybind11;

double NegBinomial_LogLikelihood(py::array_t<int> deaths_series, py::array_t<double> ag_mu_series, 
double eta){
  // set fixed elements:
  auto deaths = deaths_series.unchecked<2>(); 
  auto ag_mu = ag_mu_series.unchecked<2>(); 

  double negative_infinity = - std::numeric_limits<double>::infinity();
  double lfx = 0.0;
  for(ssize_t i=0;i<deaths.shape(0);++i)
    {
      for(ssize_t j=0;j<deaths.shape(1);++j)
        {
          double mu = ag_mu(i,j);
          int x = deaths(i,j);
          if((mu == 0) && (x == 0))
            {
              lfx += 0.0;
            }
          else if((mu == 0) & (x != 0))
            {
              lfx += negative_infinity;
            }
          else
            {
              if(eta > 1.5E-08)
                {
                  double r = mu / eta;
                  lfx += lgamma(x + r) - lgamma(r);
                  double p = 1.0 - (1.0 / (eta + 1.0));
                  lfx += ((r * log(1 - p)) + (x * log(p)));                 
                }
            }

        }

    }
  return lfx;
}
PYBIND11_PLUGIN(death_lik) {
    pybind11::module m("death_lik", "auto-compiled c++ extension");
    m.def("NegBinomial_LogLikelihood", &NegBinomial_LogLikelihood, py::return_value_policy::take_ownership);
    return m.ptr();
}
