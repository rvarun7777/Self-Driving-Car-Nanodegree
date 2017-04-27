//============================================================================
// Name        : ukf-update.cpp
// Author      : ddigges
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

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

    VectorXd x_out = VectorXd(5);
    MatrixXd P_out = MatrixXd(5, 5);
    ukf.UpdateState(&x_out, &P_out);

	return 0;
}
