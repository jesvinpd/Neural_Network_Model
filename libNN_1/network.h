#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <list> 
#include <iostream>
#include <fstream>  // Add this line
#include <string>   // Add this line

#include "neuron.h"

class network : public neuron, public input, public output, public error, public label
{
	public:
		std::list<std::list<struct node>> nodes;
		
		network () 
		{
		
		}
		
		void initialize (std::list<std::list<std::pair<ACT_FUN, double>>> node_list) 
		{
			unsigned int l_index = 0;
		
			for (auto l_id = node_list.begin (); l_id != node_list.end (); ++l_id)
			{
				std::list<struct node>  temp_layer;
			
				unsigned int p_index = 0;
			
				for (auto n_id = l_id->begin (); n_id != l_id->end (); ++n_id)
				{			
					struct node temp_node;
		
					temp_node.f = n_id->first;
					
					temp_node.b = n_id->second;
					
					temp_node.l_index = l_index;
					
					temp_node.p_index = p_index;
		
					temp_layer.push_back (temp_node);
					
					p_index++;
				}
			
				nodes.push_back (temp_layer);
				
				l_index++;
			}
		}
			

		void set_learning_rate (double _rate)
		{
			learning_rate = _rate;
		}
		
		
		
		void add (nn_to_nn &nn) 
		{
			assert (nn.to_layer < nodes.size ());
			
			assert (nn.from_layer < nodes.size ());
			
			auto it_from_layer = nodes.begin ();
			
			std::advance (it_from_layer, nn.from_layer);
			
			assert (nn.from_node < it_from_layer->size ());
			
			auto it_from_node = it_from_layer->begin ();
			
			std::advance (it_from_node, nn.from_node);
			
			
			auto it_to_layer = nodes.begin ();
			
			std::advance (it_to_layer, nn.to_layer);
			
			assert (nn.to_node < it_to_layer->size ());
			
			auto it_to_node = it_to_layer->begin ();
			
			std::advance (it_to_node, nn.to_node);
					
			
			it_from_node->w_out.push_back (std::pair (nn.trainable, nn.weight));			

			nn.from_index = it_from_node->w_out.size () - 1;	
			
			it_to_node->w_in.push_back (&(it_from_node->w_out.back ().second));
			
			nn.to_index = it_to_node->w_in.size () - 1;		
			
			it_to_node->in.push_back (&(it_from_node->out));
			
			it_from_node->error.push_back (&(it_to_node->delta));
			
			it_from_node->label.push_back (&(it_to_node->out));
			
			assert (it_from_node->label.size () == it_from_node->error.size ());
			
			assert (it_from_node->w_out.size () == it_from_node->error.size ());
			
			assert (it_to_node->w_in.size () == it_to_node->in.size ());	
		}
		
		void add (ip_to_nn &ip) 
		{
			assert (ip.to_layer < nodes.size ());
			
			auto it_layer = nodes.begin ();
			
			std::advance (it_layer, ip.to_layer);
			
			assert (ip.to_node < it_layer->size ());
			
			auto it_node = it_layer->begin ();
			
			std::advance (it_node, ip.to_node);
			
			double val = 1.0;
			
			in.push_back (val);
			
			it_node->in.push_back (&(in.back ()));
		
			w_out.push_back (std::pair (ip.trainable, ip.weight));
			
			it_node->w_in.push_back (&(w_out.back ().second));
			
			ip.to_index = it_node->w_in.size () - 1;
					
			assert (it_node->w_in.size () == it_node->in.size ());
		}
		
		
		
		void add (nn_to_op &op) 
		{
			assert (op.from_layer < nodes.size ());
			
			auto it_layer = nodes.begin ();
			
			std::advance (it_layer, op.from_layer);
			
			assert (op.from_node < it_layer->size ());
			
			auto it_node = it_layer->begin ();
			
			std::advance (it_node, op.from_node);
			
			out.push_back (&it_node->out);
			
			it_node->w_out.push_back (std::pair (op.trainable, op.weight));
			
			double val = 1.0;
			
			err.push_back (val);
			
			lbl.push_back (val);
			
			it_node->error.push_back (&(err.back ()));
			
			it_node->label.push_back (&(lbl.back ()));
			
			op.from_index = it_node->error.size () - 1;	
			
			assert (it_node->error.size () == it_node->w_out.size ());
			
			assert (it_node->error.size () == it_node->label.size ());	
		}
				
		void predict () 
		{
			for (auto l_id = nodes.begin (); l_id != nodes.end (); ++l_id)
			{
				for (auto n_id = l_id->begin (); n_id != l_id->end (); ++n_id)  
				{
					in_forward = true;
					
					forward (n_id);
					
					in_forward = false;
				}	
			}
		}	
		
		void train () 
		{
			for (auto l_id = nodes.end (); l_id != nodes.begin ();)
			{
				--l_id;
				
				for (auto n_id = l_id->begin (); n_id != l_id->end (); ++n_id)
				{	
					in_backward = true;
					
					backward (n_id);
					
					in_backward = false;
				}	
			}
		}

		void print_weight () const
		{
			for (std::list<std::list<struct node>>::const_iterator l_id = nodes.begin (); l_id != nodes.end (); ++l_id)
			{
				for (std::list<struct node>::const_iterator n_id = l_id->begin (); n_id != l_id->end (); ++n_id)  
				{				
					printf ("L-ID: %d\t,N-ID:%d\t Bias: [ %f ]\t Delta: [ %f ]", n_id->l_index, n_id->p_index, n_id->b, n_id->delta);
					
					printf ("]\nIn wights: [");
					
					for (auto index : n_id->w_in)
					{
						printf ("%f ", *index);
					}
					
					printf ("]\nOut wights: [");
					
					for (auto index : n_id->w_out)
					{
						printf ("%f ", index.second);
					}
					
					std::cout << "]" <<std::endl;
				}	
			}
		}		

		void save_weights_to_csv(const std::string& filename, unsigned int epoch) const
		{
			std::ofstream file;
			
			// Append mode - creates file if doesn't exist, appends if it does
			if (epoch == 0)
			{
				file.open(filename, std::ios::out);  // Overwrite on first epoch
				file << "Epoch,Layer,Node,WeightType,WeightIndex,WeightValue\n";
			}
			else
			{
				file.open(filename, std::ios::app);  // Append for subsequent epochs
			}
			
			if (!file.is_open())
			{
				std::cerr << "Error: Could not open file " << filename << std::endl;
				return;
			}
			
			// Iterate through all layers
			unsigned int layer_idx = 0;
			for (auto l_id = nodes.begin(); l_id != nodes.end(); ++l_id, ++layer_idx)
			{
				unsigned int node_idx = 0;
				// Iterate through all nodes in layer
				for (auto n_id = l_id->begin(); n_id != l_id->end(); ++n_id, ++node_idx)
				{
					// Save input weights
					unsigned int weight_idx = 0;
					for (auto w_it = n_id->w_in.begin(); w_it != n_id->w_in.end(); ++w_it, ++weight_idx)
					{
						file << epoch << "," 
							 << layer_idx << "," 
							 << node_idx << ","
							 << "input," 
							 << weight_idx << ","
							 << **w_it << "\n";
					}
					
					// Save output weights
					weight_idx = 0;
					for (auto w_it = n_id->w_out.begin(); w_it != n_id->w_out.end(); ++w_it, ++weight_idx)
					{
						file << epoch << "," 
							 << layer_idx << "," 
							 << node_idx << ","
							 << "output," 
							 << weight_idx << ","
							 << w_it->second << "\n";
					}
					
					// Save bias
					file << epoch << "," 
						 << layer_idx << "," 
						 << node_idx << ","
						 << "bias," 
						 << "0," 
						 << n_id->b << "\n";
				}
			}
			
			file.close();
		}
};

#endif
