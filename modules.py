import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        input_dim = kwargs.get('input_dim', 40)
        h_dim = kwargs.get('h_dim', 128)
        dropout = kwargs.get('dropout', 0.1)
        num_layers = kwargs.get('num_layers', 1)
        layer_norm = kwargs.get('layer_norm', False)
        output_dim = kwargs.get('output_dim',1)
        self.mode = kwargs.get('mode')
        
        main_stream = []

        if layer_norm:
            main_stream.append(nn.LayerNorm(input_dim))
        
        curr_output_dim = h_dim
        curr_main_input_dim = input_dim

        for _ in range(num_layers):
            main_stream.append(nn.Linear(curr_main_input_dim, curr_output_dim))
            main_stream.append(nn.ReLU())
            main_stream.append(nn.Dropout(dropout))
            curr_main_input_dim = curr_output_dim
            curr_output_dim = curr_output_dim // 2

        # manifold layer:
        curr_output_dim = 2
        self.manifold_layer = nn.Linear(curr_main_input_dim, curr_output_dim)
        curr_main_input_dim = curr_output_dim

        # By avoiding non linear activations after the manifold layer, we induce a linear geometry on manifold frontiers!
        self.output_module = nn.Sequential(
            nn.Linear(curr_main_input_dim, output_dim))
        
        self.main_stream = nn.Sequential(*main_stream)
        # print("Architecture of the MLP: ")
        # print(self.parameters)
            
    def forward(self, x):
        
        main = self.main_stream(x)
        manifold = self.manifold_layer(main)
        return self.output_module(manifold), manifold.detach()
    
    
    def initialize_weights(self, strategy='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if strategy == 'xavier':
                    # Xavier Initialization
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif strategy == 'he':
                    # He Initialization
                    init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif strategy == 'normal':
                    # Normal Initialization
                    init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                else:
                    raise ValueError(f"Unknown local initialization strategy: {strategy}")
                
