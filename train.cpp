#include <iostream>
#include <string>

#include "engine.hpp"
#include "nn.hpp"

using std::cout;
using std::endl;
using std::string;

// double loss(int batch_size) {
//     if (batch_size == -1) {

//     }
// }

int main() {
    MLP mlp(2, {16, 16, 1});

    cout << "MLP: " << mlp << endl;
    cout << "The number of parameters is: " << mlp.parameters().size() << endl;
}