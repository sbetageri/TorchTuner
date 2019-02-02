import torch
import torch.nn as nn

from collections import OrderedDict

def getSequentialNetwork(in_feat=1024, 
                         out_feat=10, 
                         hidden_layers=[], 
                         act_func=nn.ReLU(),
                         output_act_func=nn.Softmax()):
    # Merging layer dimensions into a single list
    layers = [in_feat]
    layers.extend(hidden_layers)
    layers.append(out_feat)

    seq_layers = OrderedDict()

    fc_base_name = 'fc_'
    act_base_name = 'act_'
    output_name = 'output'
    for i in range(0, len(layers) - 1):
        fc_name = fc_base_name + str(i + 1)
        act_name = act_base_name + str(i + 1)
        seq_layers[fc_name] = nn.Linear(in_features=layers[i],
                                        out_features=layers[i + 1])
        if i != len(layers) - 2:
            seq_layers[act_name] = act_func
        else:
            seq_layers[output_name] = output_act_func
    
    seq_network = nn.Sequential(layers)
    return seq_network


if __name__ == '__main__':
    hidden_layers = [512, 256, 128, 64, 32]
    layers = getSequentialNetwork(hidden_layers=hidden_layers)
    for layer in layers:
        print(layer, layers[layer])