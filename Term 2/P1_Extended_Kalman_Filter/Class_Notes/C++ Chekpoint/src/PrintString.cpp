#include <iostream>
#include <string>
#include "PrintString.h"

// it's also worth noting that `string` and `cout` live in namespace std, eg. `std::string`.
// with the declaration on the next line, you can just use `string` and `cout`.
using namespace std;

void PrintString(string str, int n)
{
	for(int i=1; i<=n; ++i) {
		std::cout << str << std::endl;
	}
}
