#include <iostream>
#include "ukf.h"

UKF::UKF() {
  //TODO Auto-generated constructor stub
  Init();
}

UKF::~UKF() {
  //TODO Auto-generated destructor stub
}

void UKF::Init() {

}

VectorXd calculate_transition(VectorXd sigma_point, double delta_t, int n_x) {
	VectorXd transition(n_x);
	transition.fill(0.0);

	double px = sigma_point(0);
	double py = sigma_point(1);
	double v = sigma_point(2);
	double psi = sigma_point(3);
	double psi_dot = sigma_point(4);
	double long_acceleration = sigma_point(5);
	double yaw_rate_acceleration =  sigma_point(6);

	VectorXd process_noise(n_x);

	process_noise(0) = (1/2.0 * delta_t * delta_t * cos(psi) * long_acceleration);
	process_noise(1) = (1/2.0 * delta_t * delta_t * sin(psi) * long_acceleration);
	process_noise(2) = (long_acceleration * delta_t);
	process_noise(3) = (1/2.0 * delta_t * delta_t * yaw_rate_acceleration);
	process_noise(4) = (yaw_rate_acceleration * delta_t);

	if(psi_dot != 0) {
		transition(0) = (v/(float)psi_dot * (sin(psi + psi_dot * delta_t) - sin(psi)));
		transition(1) = (v/(float)psi_dot * (-cos(psi + psi_dot * delta_t) + cos(psi)));
		transition(2) = 0;
		transition(3) = (psi_dot * delta_t);
		transition(4) = 0;
	} else {
		transition(0) = (v * cos(psi) * delta_t);
		transition(1) = (v * sin(psi) * delta_t);
		transition(2) = 0;
		transition(3) = (psi_dot * delta_t);
		transition(4) = 0;
	}

	return transition + process_noise;
}

VectorXd predict(VectorXd sigma_point, double delta_t, int n_x) {
	return sigma_point.head(n_x) + calculate_transition(sigma_point, delta_t, n_x);
}



void predictSigmaPoints(MatrixXd Xsig_aug, MatrixXd* Xsig_pred, double delta_t, int n_x) {
	for(int i=0; i<Xsig_aug.cols(); ++i) {
		(*Xsig_pred).col(i) = predict(Xsig_aug.col(i), delta_t, n_x);
	}
}


/*******************************************************************************
* Programming assignment functions:
*******************************************************************************/

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {

  //set state dimension
  int n_x = 5;

  //set augmented dimension
  int n_aug = 7;

  //create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
     Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  double delta_t = 0.1; //time diff in sec
/*******************************************************************************
 * Student part begin
 ******************************************************************************/

  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column

  predictSigmaPoints(Xsig_aug, &Xsig_pred, delta_t, n_x);


/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;

}
