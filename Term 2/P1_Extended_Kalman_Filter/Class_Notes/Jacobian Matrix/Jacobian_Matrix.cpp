#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main() {

	/*
	 * Compute the Jacobian Matrix
	 */

	//predicted state  example
	//px = 1, py = 2, vx = 0.2, vy = 0.4
	VectorXd x_predicted(4);
	x_predicted << 1, 2, 0.2, 0.4;

	MatrixXd Hj = CalculateJacobian(x_predicted);

	cout << "Hj:" << endl << Hj << endl;

	return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	if(px == 0 && py == 0) {
		return Hj;
	}

	float px2py2 = pow(px, 2) + pow(py, 2);

	Hj(0, 0) = px/sqrt(px2py2);
	Hj(0, 1) = py/sqrt(px2py2);

	Hj(1, 0) = -py/px2py2;
	Hj(1, 1) = px/px2py2;

	Hj(2, 0) = py* (vx*py - vy*px)/pow(px2py2, 3/2);
	Hj(2, 1) = px * (vy*px - vx*py)/pow(px2py2, 3/2);
	Hj(2, 2) = px/sqrt(px2py2);
	Hj(2, 3) = py/sqrt(px2py2);

	return Hj;
}
