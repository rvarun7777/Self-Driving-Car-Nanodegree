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

    MatrixXd Xsig = MatrixXd(11, 5);
    ukf.GenerateSigmaPoints(&Xsig);

    //print result
    std::cout << "Xsig = " << std::endl << Xsig << std::endl;

	return 0;
}
