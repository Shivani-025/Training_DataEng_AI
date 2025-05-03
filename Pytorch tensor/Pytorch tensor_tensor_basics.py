import torch
import numpy as np

# From a Python list
data_list = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data_list)
print('List Tensor \n', tensor_from_list)

# From a NumPy array
data_np = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.tensor(data_np)
print('Numpy Tensor \n', tensor_from_np)

# Using Built-in Functions:

# Creating a tensor of zeros
zeros_tensor = torch.zeros((2, 3))
print(zeros_tensor)

# Creating a tensor of ones
ones_tensor = torch.ones((2, 3))
print(ones_tensor)

# Creating a tensor with random values
rand_tensor = torch.rand((2, 3))
print(rand_tensor)



#From Existing Data:

# From a Python list
data_list = [1, 2, 3, 4]
tensor_from_list = torch.tensor(data_list)
print(tensor_from_list)

# From a NumPy array
data_np = np.array([1, 2, 3, 4])

tensor_from_np = torch.tensor(data_np)
print(tensor_from_np)


#Basic Operations (Addition, Subtraction, Multiplication, Division)

# Creating tensors
data_list = [[1, 2], [3, 4]]
data_np = np.array([[1, 2], [3, 4]])

tensor_a = torch.tensor(data_list)            #[1, 2, 3])
tensor_b = torch.tensor(data_np)               #[4, 5, 6])

# Addition
add_result = tensor_a + tensor_b
print("Addition:", add_result)

# Subtraction
sub_result = tensor_a - tensor_b
print("Subtraction:", sub_result)

# Multiplication
mul_result = tensor_a * tensor_b
print("Multiplication:", mul_result)

# Division
div_result = tensor_a // tensor_b
print("Division:", div_result)


#Reshaping Tensors (view, reshape)

# Creating a tensor
tensor_data1 = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
#tensor_data2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


# Reshaping with view (note: view requires the tensor to be contiguous in memory)
reshaped_view = tensor_data1.view(4, 2)
print("Reshaped with view:", reshaped_view)

# Reshaping with reshape (more flexible)
reshaped_reshape = tensor_data1.reshape(4, 2)
print("Reshaped with reshape:", reshaped_reshape)

# Slicing and Indexing Tensors

# Creating a tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Indexing
first_element = tensor[0, 0]
print("First element:", first_element)

# Slicing
first_row = tensor[0, :]
print("First row:", first_row)

second_column = tensor[:, 1]
print("Second column:", second_column)

# Advanced Operations (Concatenation, Stacking)
# concatenation and stacking are useful for combining tensors.

# Creating tensors
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

# Concatenation along the first dimension (rows)
concat_result = torch.cat((tensor_a, tensor_b), dim=0)
print("Concatenated along rows:", concat_result)

# Concatenation along the second dimension (columns)
concat_result = torch.cat((tensor_a, tensor_b), dim=1)
print("Concatenated along columns:", concat_result)

# Stacking (creates a new dimension)
stacked_result = torch.stack((tensor_a, tensor_b), dim=0)
print("Stacked along new dimension:", stacked_result)
