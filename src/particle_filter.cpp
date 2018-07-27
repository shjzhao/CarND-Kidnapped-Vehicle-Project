/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    // std::random_device rd;
    // std::mt19937 gen(rd());
     default_random_engine gen;
    // Creates a normal (Gaussian) distribution for x, y, theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 1000;

    for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;
        particles.push_back(particle);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    // std::random_device rd;
    // std::mt19937 gen(rd());
     default_random_engine gen;
    // Creates a normal (Gaussian) distribution for x, y, theta
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    if (fabs(yaw_rate < 0.0001)) {
        for (int i = 0; i < num_particles; ++i) {
            particles[i].x += velocity*delta_t*cos(particles[i].theta) + dist_x(gen);
            particles[i].y += velocity*delta_t*sin(particles[i].theta) + dist_y(gen);
            particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
        }
    }
    else {
        for (int i = 0; i < num_particles; ++i) {
            particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta))
                              + dist_x(gen);
            particles[i].y += velocity/yaw_rate*(cos(particles[i].theta) - cos(yaw_rate*delta_t + particles[i].theta))
                              + dist_y(gen);
            particles[i].theta += yaw_rate*delta_t + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); ++i) {
        // init minimum distance to maximum possible
        double min_dist = numeric_limits<double>::max();
        for (int j = 0; j < predicted.size(); ++j){
            double dist = sqrt((predicted[j].x - observations[i].x)*(predicted[j].x - observations[i].x)
                    + (predicted[j].y - observations[i].y)*(predicted[j].y - observations[i].y));
            if (dist < min_dist) {
                observations[i].id = predicted[j].id;
                min_dist = dist;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    double gauss_norm= 1/(2*M_PI*std_landmark[0]* std_landmark[1]);

    for (int i = 0; i < num_particles; ++i) {
        // for particles out of the sensor range, initialize the weight to 0.0
        particles[i].weight = 0.0;
        std::vector<LandmarkObs> predictions;
        for (int j =0; j < map_landmarks.landmark_list.size(); ++j) {
            LandmarkObs predict;
            if (fabs(map_landmarks.landmark_list[j].x_f - particles[i].x) < sensor_range &&
                fabs(map_landmarks.landmark_list[j].y_f - particles[i].y) < sensor_range) {
                predict.x = map_landmarks.landmark_list[j].x_f;
                predict.y = map_landmarks.landmark_list[j].y_f;
                predict.id = map_landmarks.landmark_list[j].id_i;
                predictions.push_back(predict);
                // for particles in the sensor_range, initialize the weight to 1.0
                particles[i].weight = 1.0;
            }
        }
        // observations transformed from vehicle coordinates to map coordinates
        vector<LandmarkObs> tf_observations;
        for (int j = 0; j < observations.size(); ++j) {
            LandmarkObs tf_observation;
            // transform to map x coordinate
            tf_observation.x = particles[i].x + cos(particles[i].theta) * observations[j].x
                    - sin(particles[i].theta) * observations[j].y;
            // transform to map y coordinate
            tf_observation.y = particles[i].y + sin(particles[i].theta) * observations[j].x
                    + cos(particles[i].theta) * observations[j].y;
            tf_observation.id = particles[i].id;
            tf_observations.push_back(tf_observation);
        }
        dataAssociation(predictions, tf_observations);
        // particles[i].weight = 1.0;

        for (int j = 0; j < tf_observations.size(); ++j) {
            for (int k = 0; k < predictions.size(); ++k) {
                if (tf_observations[j].id == predictions[k].id) {
                    // calculate exponent
                    double exponent = (tf_observations[j].x - predictions[k].x) * (tf_observations[j].x - predictions[k].x)
                                      / (2 * std_landmark[0] * std_landmark[0])
                                      +
                                      (tf_observations[j].y - predictions[k].y) * (tf_observations[j].y - predictions[k].y)
                                      / (2 * std_landmark[1] * std_landmark[1]);
                    particles[i].weight *= gauss_norm * exp(-exponent);
                }
            }
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> sample_particles;

    // get all of the current weights
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }
    default_random_engine gen;
    std::discrete_distribution<> disc_dist(weights.begin(), weights.end());
    for (int i = 0; i < num_particles; i++) {
        sample_particles.push_back(particles[disc_dist(gen)]);
    }
    // avoid the deep copy of vector data by using move semantics
    particles = std::move(sample_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
