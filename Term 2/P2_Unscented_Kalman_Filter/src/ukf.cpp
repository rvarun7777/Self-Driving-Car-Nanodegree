#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using std::vector;
using namespace Eigen;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.8;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  n_x_ = 5 ;

  ///* Augmented state dimension
  n_aug_= 7;

  time_us_ = 0;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2*n_aug_+1);

  R_laser_ = MatrixXd(2,2);
  R_laser_.setZero();
  R_laser_(0,0) = std_laspx_*std_laspx_;
  R_laser_(1,1) = std_laspy_*std_laspy_;

  R_radar_ = MatrixXd(3,3);
  R_radar_.setZero();
  R_radar_(0,0) = std_radr_*std_radr_;
  R_radar_(1,1) = std_radphi_*std_radphi_;
  R_radar_(2,2) = std_radrd_*std_radrd_;

  Tc_laser_ = MatrixXd(n_x_, 2);

  Tc_radar_ = MatrixXd(n_x_, 3);

  Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  stochaistic = VectorXd(5);
  deterministic = VectorXd(5);
  Zsig_laser = MatrixXd(2 , 2 * n_aug_ + 1);
  Zsig_radar = MatrixXd(3 , 2 * n_aug_ + 1);

  S_laser = MatrixXd(2 , 2);
  S_radar = MatrixXd(3 , 3);

  //create augmented mean vector
  x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  z_pred_laser = VectorXd(2);
  z_pred_radar = VectorXd(3);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      x_ << convertToCartesian(meas_package.raw_measurements_);

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    P_ <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
        -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
        0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
        -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
        -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  generateSigmaPoints();
  augmentSigmaPoints();
  predictSigmaPoints(delta_t);
  predictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  n_z_ = 2;
  R_ = R_laser_;

  VectorXd z = meas_package.raw_measurements_;

  MatrixXd Zsig = Zsig_laser;

  //mean predicted measurement
  z_pred_ = z_pred_laser;

  //measurement covariance matrix S
  S_ = S_laser;

  //transform sigma points into measurement space
  Zsig.setZero();
  for (int i =0; i < Zsig.cols(); i++) {
    double px = Xsig_pred_.col(i)[0];
    double py = Xsig_pred_.col(i)[1];
    Zsig.col(i) << px, py;
  }
  //calculate mean predicted measurement

  z_pred_.setZero();
  for (int i =0; i < Zsig.cols(); i++) {
    z_pred_ += weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  S_.setZero();
  for (int i =0; i < Zsig.cols(); i++) {
    MatrixXd col = Zsig.col(i)-z_pred_;
    S_ += weights_(i) * col * col.transpose();
  }
  // Adding Measurement covariance
  S_ += R_;

  //calculate cross correlation matrix
  Tc = Tc_laser_;
  Tc.setZero();
  for (int i = 0 ; i< 2 * n_aug_ + 1; i++ ) {
    MatrixXd x_space = Xsig_pred_.col(i) -x_;
    MatrixXd z_space = Zsig.col(i) -z_pred_;
    Tc  += weights_(i) * x_space* z_space.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  //update state mean and covariance matrix
  MatrixXd z_diff = z - z_pred_;
  x_ = x_ + K *z_diff;
  P_ = P_ - K * S_ * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  n_z_ = 3;
  R_ = R_radar_;
  VectorXd z = meas_package.raw_measurements_;

  MatrixXd Zsig = Zsig_radar;

  //mean predicted measurement
  z_pred_ = z_pred_radar;

  //measurement covariance matrix S
  S_ = S_radar;

  //transform sigma points into measurement space
  Zsig.setZero();
  for (int i =0; i < Zsig.cols(); i++) {
    double px = Xsig_pred_.col(i)[0];
    double py = Xsig_pred_.col(i)[1];
    double v = Xsig_pred_.col(i)[2];
    double yaw = Xsig_pred_.col(i)[3];
    double sqrpxpy = pow(px,2) + pow(py,2);
    double sqrtpxpy = sqrt(sqrpxpy);
    if (px!=0 && sqrtpxpy > 0.001){
      Zsig.col(i) << sqrtpxpy, atan2(py,px), (px*v*cos(yaw) + py*v*sin(yaw))/sqrtpxpy;
    } else {
      Zsig.col(i) << sqrtpxpy, 0, 0;
    }

  }
  //calculate mean predicted measurement

  z_pred_.setZero();
  for (int i =0; i < Zsig.cols(); i++) {
    z_pred_ += weights_(i) * Zsig.col(i);

    //angle normalization
    while (z_pred_(1)> M_PI) z_pred_(1)-=2.*M_PI;
    while (z_pred_(1)<-M_PI) z_pred_(1)+=2.*M_PI;
  }

  //calculate measurement covariance matrix S

  S_.setZero();
  for (int i =0; i < Zsig.cols(); i++) {
    MatrixXd col = Zsig.col(i)-z_pred_;
    while (col(1)> M_PI) col(1)-=2.*M_PI;
    while (col(1)<-M_PI) col(1)+=2.*M_PI;
    S_ += weights_(i) * col * col.transpose();
  }
  // Adding Measurement covariance
  S_ += R_;

  //calculate cross correlation matrix
  Tc = Tc_radar_;
  Tc.setZero();

  for (int i = 0 ; i< 2 * n_aug_ + 1; i++ ) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred_;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  //update state mean and covariance matrix
  MatrixXd z_diff = z - z_pred_;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  x_ = x_ + K *z_diff;
  P_ = P_ - K * S_ * K.transpose();
}

void UKF::generateSigmaPoints() {
  //define spreading parameter
  lambda_ = 3 - n_x_;
  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //calculate sigma points ...
  //set sigma points as columns of matrix Xsig
  Xsig.setZero();
  Xsig.col(0) = x_ ;
  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_+n_x_) * A.col(i);
  }
}

void UKF::predictSigmaPoints(double delta_t) {

  double half_delta_t2 = delta_t*delta_t/2;
  Xsig_pred_.setZero();
  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column

  for (int i =0 ; i< Xsig_aug.cols(); i++) {
    double velocity = Xsig_aug.col(i)[2];
    double yaw = Xsig_aug.col(i)[3];
    double yawdot = Xsig_aug.col(i)[4];
    double noise_long = Xsig_aug.col(i)[5];
    double noise_yaw = Xsig_aug.col(i)[6];
    double yawdot_times_delta_t = yawdot*delta_t;

    stochaistic << half_delta_t2*cos(yaw)*noise_long,
        half_delta_t2*sin(yaw)*noise_long,
        delta_t*noise_long,
        half_delta_t2*noise_yaw,
        delta_t*noise_yaw;


    if (fabs(yawdot) > 0.0001) {
      double velocity_by_yawdot = velocity/yawdot;
      deterministic << velocity_by_yawdot*(sin(yaw + yawdot_times_delta_t) -sin(yaw)),
          velocity_by_yawdot*(-cos(yaw + yawdot_times_delta_t) +cos(yaw)),
          0,
          yawdot_times_delta_t,
          0;

      Xsig_pred_.col(i) = Xsig_aug.col(i).topRows(5) + deterministic + stochaistic ;
    }
    else {
      deterministic << velocity*cos(yaw)*delta_t,
          velocity*sin(yaw)*delta_t,
          0,
          yawdot_times_delta_t,
          0;

      Xsig_pred_.col(i) = Xsig_aug.col(i).topRows(5) + deterministic + stochaistic;
    }

  }
}

void UKF::augmentSigmaPoints() {
  //define spreading parameter
  lambda_ = 3 - n_aug_;

  Xsig_aug.setZero();

  //create augmented mean state
  x_aug << x_,
      0,
      0;
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_*std_a_, 0,
      0, std_yawdd_*std_yawdd_;

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;
  //create square root matrix
  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  for (int i =0; i< n_aug_; i++) {
    Xsig_aug.col(i+1) = x_aug + P_aug_sqrt.col(i) * sqrt(lambda_ + n_aug_);
    Xsig_aug.col(i+1+n_aug_) = x_aug - P_aug_sqrt.col(i) * sqrt(lambda_ + n_aug_);
  }

}

Eigen::VectorXd UKF::convertToCartesian(Eigen::VectorXd matrix) {
  VectorXd z = VectorXd(5);
  z << matrix[0] * cos(matrix[1]), matrix[0] * sin(matrix[1]), 0, matrix[1], 0;
  return z;
}

void UKF::predictMeanAndCovariance() {
  //set weights
  double common_weight = 1/((lambda_ + n_aug_)*2);
  double first_weight = lambda_/(lambda_ + n_aug_);
  weights_.fill(common_weight);
  weights_(0) = first_weight;

  //predict state mean
  x_.setZero();
  for (int i =0; i < 2 *n_aug_+1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  //predict state covariance matrix
  P_.setZero();
  for (int i =0; i < 2 *n_aug_+1; i++) {
    MatrixXd col = Xsig_pred_.col(i)-x_;
    while (col(3)> M_PI) col(3)-=2.*M_PI;
    while (col(3)<-M_PI) col(3)+=2.*M_PI;
    P_ += weights_(i) * col* col.transpose();
  }



}