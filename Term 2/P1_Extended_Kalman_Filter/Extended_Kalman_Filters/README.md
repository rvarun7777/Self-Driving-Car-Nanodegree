# Extended Kalman Filter by C++
Extended Kalman Filter(EKF) implementation by C++.  

[//]: # (Image References)
[image1]: ./image_for_README/file1.png
[image2]: ./image_for_README/file2.png

## Output

| RMSE | output 1 |　output 2  |
|:----:|:----:|:----:|
| position x | 0.0651795 | 0.183608 |
| position y | 0.0605726 | 0.190305 |
| speed x | 0.544212 | 0.499585 |
| speed y|  0.544226 | 0.805314 |

![alt text][image1]
![alt text][image2]



## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && cd .. && make`
4. Run it: `./ExtendedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`
