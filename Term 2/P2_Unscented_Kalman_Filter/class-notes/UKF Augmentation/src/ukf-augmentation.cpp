#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "ukf.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

int main() {

	//Create a UKF instance
	UKF ukf;

/*******************************************************************************
* Programming assignment calls
*******************************************************************************/

    MatrixXd Xsig_aug = MatrixXd(15, 7);
    ukf.AugmentedSigmaPoints(&Xsig_aug);

	return 0;
}
