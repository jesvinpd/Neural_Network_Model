#ifndef __MODEL_H__
#define __MODEL_H__

#include "network.h"

class model
{
	public:
		network _nn = network ();
	
		model ()
		{
			
		}
		
		void set ()
		{
			std::list<std::list<std::pair<ACT_FUN, double>>> node_list;
			
			node_list = {{{ACT_FUN::leaky_relu, 4.0}}, {{ACT_FUN::leaky_relu, 5.0}, {ACT_FUN::leaky_relu, 2.0}, {ACT_FUN::leaky_relu, 4.0}}, {{ACT_FUN::leaky_relu, 3.0}}};
			
			_nn.initialize (node_list);		
			
			_nn.set_learning_rate (0.0000000002);


			ip_to_nn i1 = {0, 0, 0, 1.0, false};
			
			ip_to_nn i2 = {0, 0, 0, -1.0, false};
			
		
			_nn.add (i1);
			
			_nn.add (i2);
			

			nn_to_nn n0n10 = {0, 0, 0, 1, 0, 0, 1.0, false};
			
			nn_to_nn n0n11 = {0, 0, 0, 1, 1, 0, 1.0, false};
			
			nn_to_nn n0n12 = {0, 0, 0, 1, 2, 0, 1.0, false};
			
			nn_to_nn n10n20 = {1, 0, 0, 2, 0, 0, 1.0, true};
			
			nn_to_nn n11n20 = {1, 1, 0, 2, 0, 0, 1.0, true};
			
			nn_to_nn n12n20 = {1, 2, 0, 2, 0, 0, 1.0, true};
			
			nn_to_nn n10n10 = {1, 0, 0, 1, 0, 0, 1.0, false};
			
			nn_to_nn n12n11 = {1, 2, 0, 1, 1, 0, -1.0, false};
		
		
			_nn.add (n0n10);
			
			_nn.add (n0n11);
			
			_nn.add (n0n12);
			
			_nn.add (n10n20);
			
			_nn.add (n11n20);
			
			_nn.add (n12n20);
			
			_nn.add (n10n10);
			
			_nn.add (n12n11);
		
		
			nn_to_op o1 = {2, 0, 0, 1.0, false};

			_nn.add (o1);
		}
};

#endif
