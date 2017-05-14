#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

#include "particle_filter.h"

using namespace std;

const int     kNumParticles   = 300;
const bool    kDebug          = false;
const string  kReportFileName = "output/particles%d.txt";
int           kReportCounter  = 0;

/**
 * Calculates value of 2d gaussian.
 */
double CalcGauss(const double dx, const double dy, const double sigma_x, const double sigma_y)
{
    double exp_expression = -0.5 * ((dx * dx / (sigma_x * sigma_x)) + (dy * dy / (sigma_y * sigma_y)));
    return exp(exp_expression) /(2 * M_PI * sigma_x * sigma_y);
}

/**
 * Calculates distance by coordinates differences.
*/
double CalcDist(const double dx, const double dy)
{
    return sqrt(dx * dx + dy * dy);
}

/**
 * Creates kNumParticles particles with given coordinates and adds white noise to them
 */
void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // initialize normal distribution generators
    default_random_engine gen;
    normal_distribution<double> N_x_init(x, std[0]);
    normal_distribution<double> N_y_init(y, std[1]);
    normal_distribution<double> N_theta_init(theta, std[2]);

    // initialize number of particles
    num_particles = kNumParticles;

    // initialie particles
    for(int i = 0; i < num_particles; i++)
    {
        Particle p;
        p.id = i;
        p.x = N_x_init(gen);
        p.y = N_y_init(gen);
        p.theta = N_theta_init(gen);
        p.weight = 1;

        particles.push_back(p);
    }

    is_initialized = true;
}

/**
 * Calculates new coordinates for each particle according to bicycle model.
 * Adds noise to the result coordinates.
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // initialize normal distribution generators
    default_random_engine gen;
    normal_distribution<double> N_x_init(0, std_pos[0]);
    normal_distribution<double> N_y_init(0, std_pos[1]);
    normal_distribution<double> N_theta_init(0, std_pos[2]);

    for(auto &p : particles)
    {
        // apply motion equations for each particle
        if(fabs(yaw_rate) > 1e-5)
        {
            p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta));
            p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + (yaw_rate * delta_t)));
        }
        else
        {
            p.x = p.x + velocity * delta_t * cos(p.theta);
            p.y = p.y + velocity * delta_t * sin(p.theta);
        }

        p.theta = p.theta + (yaw_rate * delta_t);

        // add some noise
        p.x += N_x_init(gen);
        p.y += N_y_init(gen);
        p.theta += N_theta_init(gen);
    }
}

/**
 * Recieves arrays of landmarks that can be measured and landmarks that were observed.
 * For each observed landmark finds the closest predicted landmark and memorizes its index id observation id field,
 * so the predicted landmark could be easily found later.
 * @param predicted - array of map landmarks which physically can be observed by the particle (global coordinates)
 * @param observations - array of observed landmarks (global coordinates)
 */
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
    // need this so every decent value would be lower than this
    double max_double_value = numeric_limits<double>::max();

    // for each observation find the closest landmark from the list of predicted landmarks
    // and memorize its index in the array of predicted landmarks as observation id.
    for(auto &o : observations)
    {
        double min_dist = max_double_value;

        for(int i = 0; i < predicted.size(); i++)
        {
            LandmarkObs &p = predicted[i];

            double dist = CalcDist(p.x - o.x, p.y - o.y);
            if(dist < min_dist)
            {
                //memorize the closest landmark's index in the array of predicted landmarks
                o.id = i;
                min_dist = dist;
            }
        }
    }
}

/**
 * Updates weights of particles for resampling.
 *
 * For each particle:
 * 1. Transform observations coordinates from particle's coordinates system to the global one
 * 2. Find landmarks on the map that can be observed by the particle considering sensor range
 * 3. For each observation find the closest possible landmark that can be observed.
 * 4. Calculate probability of all observations being true. Probability of each observation is
 *    gaussian where mean is landmark coordinates and argument is observation coordinates.
 * 5. Consider calculated probability as particle's weight.
 *
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		                            std::vector<LandmarkObs> observations, Map map_landmarks)
{

    // write particles to the file if needed
    if(kDebug)
    {
        char buf[50];
        sprintf(buf, kReportFileName.c_str(), kReportCounter++);
        write(buf);
    }

    // clear array of particles weights, we gonna fill it below
    weights.clear();

    // for each particle
    for(auto &p : particles)
    {
        // - consider the current particle correctly represents the object we are tracking,
        //   in this case array of observations should be in the current particle's coordinates system
        //
        // - transform each observation from the current particle coordinate system to the global one.

        vector<LandmarkObs> global_observations;
        for (auto &o : observations)
        {
            LandmarkObs global_obs_landmark;

            // we don't know what landmark it is yet, so set it to never used value
            global_obs_landmark.id = -1;

            // rotate and shift
            global_obs_landmark.x = o.x * cos(p.theta) - o.y * sin(p.theta) + p.x;
            global_obs_landmark.y = o.x * sin(p.theta) + o.y * cos(p.theta) + p.y;

            global_observations.push_back(global_obs_landmark);
        }

        // find landmarks that the current particle can measure, i.e. landmarks that are closer than
        // sensor range
        vector<LandmarkObs> predicted_landmarks;
        for (auto &l : map_landmarks.landmark_list)
        {
            if (CalcDist(l.x_f - p.x, l.y_f - p.y) <= sensor_range)
            {
                LandmarkObs landmark;
                landmark.id = l.id_i;
                landmark.x = l.x_f;
                landmark.y = l.y_f;
                predicted_landmarks.push_back(landmark);
            }
        }

        double w;
        // somehow we need to handle situation when there are impossible predictions or no predictions at all
        // when there should be predictions. This situation doesn't happen in the test data, but probably
        // we need to assign some small value to such an event.
        if((predicted_landmarks.size() == 0 && global_observations.size() > 0) ||
           (global_observations.size() == 0 && predicted_landmarks.size() > 0))
        {
            w = 1e-20;
        }
        else
        {
            w = 1;
            // find the closest landmark for each observation
            // landmark index in the predicted_landmarks array is stored in particles' id value after this method.
            dataAssociation(predicted_landmarks, global_observations);

            // find probability of all observations using gaussian distribution
            for (auto &o: global_observations)
            {
                // in case assosiated landmark was found
                if (o.id >= 0)
                {
                    LandmarkObs &l = predicted_landmarks[o.id];
                    w *= CalcGauss(l.x - o.x, l.y - o.y, std_landmark[0], std_landmark[1]);
                }
            }
        }

        // memorize particle weight
        p.weight = w;

        // add the weight to the array that will be used in resampling
        weights.push_back(w);
    }
}

/**
 * Creates a new list of particles using weighted sampling from existing particles with replacement.
 */
void ParticleFilter::resample()
{
    // prepare weighted discrete distribution
    default_random_engine gen;
    discrete_distribution<> dd(weights.begin(), weights.end());

    // sample num_particles particles from weighted discrete distribution
    vector<Particle> new_particles;
    for(int i = 0; i < num_particles; i++)
    {
        new_particles.push_back(particles[dd(gen)]);
    }

    particles = new_particles;
}

/**
 * Writes particles info to file. One row for a particle.
 */
void ParticleFilter::write(std::string filename)
{
	// You don't need to modify this file.

    remove(filename.c_str());
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i)
    {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " " << particles[i].id << "\n";
	}
	dataFile.close();
}
