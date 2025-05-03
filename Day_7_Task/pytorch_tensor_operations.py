'''
Assignment 1: Creating and Manipulating PyTorch Tensors

Problem 1: Create a 3x3 tensor with random values and perform the following operations:

Add 10 to each element.
Multiply each element by 2.
Calculate the mean and standard deviation of the tensor.

Problem 2: Create two tensors of size 3x3 with random values and perform element-wise addition and multiplication.

Problem 3: Create a tensor with values from 1 to 16 and reshape it to a 4x4 tensor. Extract the first two rows and last two columns.
'''


import torch

#Problem 1: Create a 3x3 tensor with random values and perform the following operations:
tensor1 = torch.rand((3,3))
print("a 3x3 tensor with random values : \n",tensor1)

#Add 10 to each element.
add_ten = tensor1 + 10
print("\nAdd ech element of tensor by 10 : \n",add_ten)

#Multiply each element by 2.
mul_two = tensor1 * 2
print("\nMultiply each element of tensor by 2 : \n",mul_two)

#Calculate the mean and standard deviation of the tensor.
tensor_mean = tensor1.mean()
print("\nMean of a given tensor : \n",tensor_mean)

tensor_std = tensor1.std()
print("\nStandard deviation of a given tensor : \n",tensor_std)

#Problem 2: Create two tensors of size 3x3 with random values and perform element-wise addition and multiplication.
tensor_a = torch.rand((3,3))
tensor_b = torch.rand((3,3))

print(tensor_a)
print(tensor_b)

add_tensors = tensor_a + tensor_b
print("\nAddition of given tensor : \n",add_tensors)


mul_tensors = tensor_a * tensor_b
print("\nMultiplication of given tensor : \n",mul_tensors)


#Problem 3: Create a tensor with values from 1 to 16 and reshape it to a 4x4 tensor. Extract the first two rows and last two columns.
t1 = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
tensor_reshape = t1.reshape(4, 4)
print("\nReshape of given tensor : \n", tensor_reshape)

frst_two_row = t1[0:2,:]
print("\nFirst two rows : \n",frst_two_row)
lst_two_col = t1[:,2:4]
print("\nLast two columns : \n",lst_two_col)