#include <chrono>
#include <unistd.h>

#include "data_set.h"
#include "training.h"
#include "model.h"

	int main ()
	{		
		data_set dts = data_set ();
		
		model mdl = model ();
		
		mdl.set ();	
		
		//nn1.print_weight ();	

		//std::list<double> _input = {2974.600000, -1919.200000, 5257.800000};
		
		//std::list<double> _label = {9537.800000, -0.600000, -4277.600000};


		std::list<double> _label = dts.get_list (3, -10000, 10000, 100000);
		
		std::list<double> _input = dts.get_list (3, -10000, 10000, 100000);

		double delta = 0.5;
		
		training obj = training ();
		
		//bool result = obj.train (nn1, 200000, v_input, v_label, delta); 
		
		bool result = obj.train (mdl._nn, _input, _label, delta); 
		
		if (result) 
		{
			std::cout << "Training successful" <<std::endl;
			
		} else {
		
			std::cout << "Try again" <<std::endl;
		}

		return 0;
	}	
	
