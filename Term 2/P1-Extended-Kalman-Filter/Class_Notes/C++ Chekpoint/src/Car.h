/*
 * Car.h
 *
 *  Created on: Mar 7, 2017
 *      Author: ddigges
 */

#ifndef CAR_H_
#define CAR_H_

class Car {
public:
	Car();
	void wearAndTear();
	bool drive();
	void fix();
private:
	bool in_working_condition_;

};

#endif /* CAR_H_ */
