#include "PID.h"


/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->p_error = 0.0;

  this->Kd = Kd;
  this->d_error = 0.0;

  this->Ki = Ki;
  this->i_error = 0.0;
}

void PID::UpdateError(double cte) {
  d_error = cte - p_error; // error diff, PID Control > 8
  p_error = cte;           // error p, PID Control > 8
  i_error = i_error + cte; // all, tau_i, PID Control > 12
}

double PID::TotalError() {
  // see PID Control > 11
  //     -tau_p * CTE     - tau_d * diff_CTE  - tau_i * int_CTE
  return -Kp    * p_error - Kd    * d_error   - Ki    * i_error;
}

