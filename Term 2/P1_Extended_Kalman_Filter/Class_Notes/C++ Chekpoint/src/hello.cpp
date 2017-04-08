//============================================================================
// Name        : cpp-primer.cpp
// Author      : Deborah Digges
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "PrintString.h"
#include "Factorial.h"
#include "Car.h"
#include "Doubler.h"

int main() {
	std::cout << "hello" << std::endl;

	// Test PrintString method
	PrintString("no more steering wheels", 5);

	// Test Factorial method
	std::cout << Factorial(4) << std::endl;

	Car car;

	// try to drive 10 times
	for (int i = 0; i < 10; i++) {
		bool didDrive = car.drive();
		if (!didDrive) {
			// car is broken! must fix it
			car.fix();
		}
	}

	int value = 25;

	std::cout << "Original value: " << value << std::endl;

	Doubler(value);

	std::cout << "Doubled value: " << value << std::endl;
	return 0;
}
