import torch
import torch.nn as nn
import numpy as np

class Reactify(nn.Module):
    def __init__(self, length: int, out_dim=4, n_filters_1=8, n_filters_2=16, n_filters_3=8, 
                        filter_shape_1=64, filter_shape_2=8, filter_shape_3=(7, 3), 
                        dropout_1=0.6, dropout_2=0.6, continuous=False):
        super().__init__()
        self.n_filters_3 = n_filters_3
        self.continuous = continuous
        
        self.intake_layers = nn.Sequential(
                                    nn.Dropout(p=dropout_1),
                                    nn.Conv1d(in_channels=1, out_channels=n_filters_1, kernel_size=filter_shape_1,),          # out_feats = length - filter_shape_1 + 1
                                    nn.MaxPool1d(kernel_size=4),                                                              # out_feats //=  4
                                    nn.Conv1d(in_channels=n_filters_1, out_channels=n_filters_2, kernel_size=filter_shape_2), # out_feats -= filter_shape_2 + 1
                                    nn.MaxPool1d(kernel_size=4)                                                               # out_feats //=  4
                                    ) 
        
        self.conv2d_block = nn.Sequential(
                                    nn.Dropout(p=dropout_1),
                                    nn.Conv2d(in_channels=n_filters_2, out_channels=n_filters_3, kernel_size=filter_shape_3, padding='same'), # out_feats stays the same
                                    nn.MaxPool2d(kernel_size=(8,2)),                                                                          # out_feats[0] //= 8, out_feats[1] //= 2, 
                                    )
        
        self.final_linear_block = nn.Sequential(
                                    nn.Dropout(p=dropout_2),
                                    nn.Flatten(),           # flatten to get (batch_size x in_feats) shape tensor
                                    nn.Linear(in_features=n_filters_3*((length+8-(filter_shape_1+4*filter_shape_2))//128), out_features=8),  
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout_2),
                                    nn.Linear(in_features=8, out_features=8),
                                    nn.ReLU(),
                                    nn.Linear(in_features=8, out_features=out_dim),
                                    # nn.Softmax(dim=1), #  if not continuous else nn.Linear(in_features=out_dim, out_features=out_dim)
                                    )
        self.continuous_output_layer = nn.Sequential(
                                    nn.Linear(in_features=out_dim, out_features=1),
                                    nn.Sigmoid()
                                    )
    
    def forward(self, inputs):
        def forward_(input1, input2):
            sig1 = self.intake_layers(input1)
            # print(f"sig1 shape {sig1.shape}")
            
            sig2 = self.intake_layers(input2)
            combined = torch.concat([   sig1[..., np.newaxis ],
                                        sig2[..., np.newaxis ], ], axis=-1)
            # print(f"combined shape {combined.shape}")
            result = self.conv2d_block(combined).squeeze()  # remove extra last dimension 
            # print(f"conv2d block result shape {result.shape}")
            result = self.final_linear_block(result)
            # print(f"final linear block result shape {result.shape}")
            return result
        
        input1 = inputs[:, 0, :][:, np.newaxis, :]    # normalized real part sum of the reactant spectra
        input2 = inputs[:, 1, :][:, np.newaxis, :]    # normalized real part of the reaction spectrum
        # print(f"input1 & input2 shape {input1.shape}")
        
        output1 = forward_(input1, input2)  # before (reagents) & after rxn (mixture) 
        output2 = forward_(input1, input1)  # reagents only -> trained to produce zeros only  
        output3 = forward_(input2, input2)  # mixture only  -> trained to produce zeros only  

        return [output1, output2, output3]