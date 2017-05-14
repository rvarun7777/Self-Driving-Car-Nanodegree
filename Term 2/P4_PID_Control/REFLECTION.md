# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

# PID reflection

- P is called the proportional term. In this project it had the biggest impact on how agile the steering and throttle reacted on the input from the simulator.
- I is the integral and sums up all errors from each loop. It compensates and evens out if the PD bias get too large
- D stands for the derivative term and smoothens the behavior of the steering and throttling to prevent overshooting.

My solution for the initial values:
- Steering > 0.16, 0.0001, 3.0
- Throttle > 0.8, 0.0001, 3.0

The PID behavior is pretty much in line with what we learned in the course content. I didn't feel like the derivative term had such a big impact, mostly in harder turns.
The proportional term though had a huge impact and basically contributed the most to the final solution. I did have a quick look at the twiddle parameter tuning as suggested in the course but this particular solution is based on manual trial and error.
I found some discussion here which helped me during testing initially until I drifted into the final values: https://robotics.stackexchange.com/questions/167/what-are-good-strategies-for-tuning-pid-loops.

