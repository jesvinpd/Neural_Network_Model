#include <chrono>
#include <unistd.h>
#include <list>
#include <iostream>

#include "data_set.h"
#include "training.h"
#include "model.h"

// Define a structure to hold input and label lists
struct Dataset
{
    std::list<double> input;
    std::list<double> label;
};

int main()
{
    data_set dts = data_set();
    model mdl = model();
    mdl.set();

    // Create a list of datasets
    std::list<Dataset> datasets = {
        {{4, 10, 16}, {22, 28, 34}},
        {{1, 2, 3}, {6, 8, 10}},
        {{5, 7, 9}, {15, 21, 27}}
        // Add more datasets as needed
    };

    double delta = 0.1; // Threshold for the error
    training obj = training();

    // Iterate over each dataset and train the model
    for(int i=0;i<30;i++){    //epochs
        printf("\n========== Epoch %d ==========\n", i);
        
        for (auto &dataset : datasets)
        {
            // Log weight before training
            printf("Epoch %d - Before: i1 weight = %f\n", i, mdl.get_i1_weight());
            bool result = obj.train(mdl._nn, dataset.input, dataset.label, delta);

            if (result)
            {
                std::cout << "Training successful" << std::endl;
            }
            else
            {
                std::cout << "Try again" << std::endl;
            }
        }
        
        // Save all weights to CSV after each epoch
        mdl._nn.save_weights_to_csv("weights_log.csv", i);
        printf("Weights saved to weights_log.csv for epoch %d\n", i);
        
        std::cout << "Completed iteration: " << i << std::endl;
    }

    return 0;
}