#include "commons.h"
#include <iostream>

using namespace std;

namespace commons {
void counter() {
  static int c = 0;
  cout << c++ << endl;
}

} // namespace commons
