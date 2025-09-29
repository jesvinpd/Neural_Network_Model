#include <chrono>
#include <unistd.h>

#include "data_set.h"
#include "training.h"
#include "model.h"

int main()
{
    data_set dts = data_set();

    model mdl = model();

    mdl.set();

    // nn1.print_weight ();

    // std::list<double> _input = {2974.600000, -1919.200000, 5257.800000};

    // std::list<double> _label = {9537.800000, -0.600000, -4277.600000};

    std::list<double> _input = {4, 10, 16};
    std::list<double> _label = {22, 28, 34};

    double delta = 0.001;//threshold for the error

    training obj = training();

    // bool result = obj.train (nn1, 200000, v_input, v_label, delta);
for(int i=0;i<30;i++){
    bool result = obj.train(mdl._nn, _input, _label, delta);

    if (result)
    {
        std::cout << "Training successful" << std::endl;
    }
    else
    {

        std::cout << "Try again" << std::endl;
    }
    std::cout << "in iteration: " << i << std::endl;
}
    return 0;
}