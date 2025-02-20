import torch
import torch.nn as nn
import torch.nn.functional as F

# class SimpleModel(nn.Module):
#     def __init__(self, input_features, hidden_dim1, hidden_dim2, output_features):
#         super(SimpleModel, self).__init__()
#         # Define the first linear layer
#         self.linear1 = nn.Linear(input_features, hidden_dim1)
#         # Define the second linear layer
#         self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
#         # Define the third linear layer
#         self.linear3 = nn.Linear(hidden_dim2, output_features)

#     def forward(self, x):
#         # Forward pass through the first linear layer
#         x = self.linear1(x)
#         # Forward pass through the second linear layer
#         x = self.linear2(x)
#         # Forward pass through the third linear layer
#         x = self.linear3(x)
#         # Apply softmax to the final output
#         return F.softmax(x, dim=-1)



class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



# example_input = torch.randn(1, 100)
# print(example_input)
# print("-------")
# example_input = torch.randn(2, 100)
# print(example_input)

# input_features=100, hidden_dim1=64, hidden_dim2=32, output_features=10

# # Example of using the model
# # Assuming input_features=100, hidden_dim1=64, hidden_dim2=32, output_features=10
# model = SimpleModel(input_features=100, hidden_dim1=64, hidden_dim2=32, output_features=10)

# # Example input tensor (batch size of 1 with input_features features)
# example_input = torch.randn(1, 100)

# # Forward pass through the model
# output = model(example_input)

# print(output)
