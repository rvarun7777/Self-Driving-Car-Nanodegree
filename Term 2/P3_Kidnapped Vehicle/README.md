# Kidnapped Vehicle Project
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[visualization]: ./resources/visualization.gif


# Project Introduction
Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

In this project you will implement a 2 dimensional particle filter in C++. Your particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step your filter will also get observation and control data. 

## Running the Code
```
> ./clean.sh
> ./build.sh
> ./run.sh
```
If everything worked you should see something like the following output:
```
Time step: 2444
Cumulative mean weighted error: x .1 y .1 yaw .02
Runtime (sec): 38.187226
Success! Your particle filter passed!
```

Your job is to build out the methods in `particle_filter.cpp` until the last line of output says:

```
Success! Your particle filter passed!
```

# Implementing the Particle Filter

The only file you should modify is `particle_filter.cpp` in the `src` directory.  

## Inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory. 

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

> * Map data provided by 3D Mapping Solutions GmbH.


#### Control Data
`control_data.txt` contains rows of control data. Each row corresponds to the control data for the corresponding time step. The two columns represent
1. vehicle speed (in meters per second)
2. vehicle yaw rate (in radians per second)

#### Observation Data
The `observation` directory includes around 2000 files. Each file is numbered according to the timestep in which that observation takes place. 

These files contain observation data for all "observable" landmarks. Here observable means the landmark is sufficiently close to the vehicle. Each row in these files corresponds to a single landmark. The two columns represent:
1. x distance to the landmark in meters (right is positive) RELATIVE TO THE VEHICLE. 
2. y distance to the landmark in meters (forward is positive) RELATIVE TO THE VEHICLE.

> **NOTE**
> The vehicle's coordinate system is NOT the map coordinate system. Your 
> code will have to handle this transformation.

## Success Criteria

1. **Accuracy**: your particle filter should localize vehicle position and yaw to within the values specified in the parameters `max_translation_error` and `max_yaw_error` in `src/main.cpp`.
2. **Performance**: your particle filter should complete execution within the time specified by `max_runtime` in `src/main.cpp`.

# Results

The minimum number of particles to pass the test is 11.
```
Cumulative mean weighted error: x 0.160364 y 0.128311 yaw 0.00511329
Runtime (sec): 0.585984
Success! Your particle filter passed!
```

The maximum number of particles to pass the test on my computer is approximately 1600.
```
Cumulative mean weighted error: x 0.107437 y 0.0998683 yaw 0.00349442
Runtime (sec): 44.2359
Success! Your particle filter passed!

```

The submitted version uses 300 particles and shows almost the same results.

```
Cumulative mean weighted error: x 0.109366 y 0.102328 yaw 0.00356961
Runtime (sec): 8.40976
Success! Your particle filter passed!
```


![alt_text][visualization]

# Reflections

* Particle filter seems nice and easy to implement.
* It looks quite effective as it starts showing good result with a small number of particles.
* Error situations were not covered in the project e.g. when a sensor blacks out for a short period.
* But it looks like some error situations can be handled quite easily by just restarting 
the filter.