import numpy as np
import matplotlib.pyplot as plt

# Define input and target vectors
x = np.array([[-2, -1, 0, 1],  # input 1
              [-1,  0, 1, 2]]) # input 2

t = np.array([-1.5, -1, 1, 1.5]) # target

s = x.shape[1]         # number of samples

# Forward propagation variables
zini = np.zeros(s) # zin for each data point
zi = np.zeros(s)   # zi for each data point
yi = np.zeros(s)   # output without training

# Weights sand biases
wi1=0.0651    # first weight of the input layer
wi2=-0.6970   # second weight of the input layer
wo=-0.1342    # first weight of the output layer
bi=0          # input bias
bo=-0.5481    # output bias

# Training variables
q=500         # training epochs
a=0.01        # learning rate



# Display the inputs and corresponding targets
print(f"This network has {x.shape[1]} inputs:")

for i in range(x.shape[1]):
    print(f"    [{x[0, i]}, {x[1, i]}] with target {t[i]}")

i = np.arange(1, 5)   # x-axis for plotting (1 to 4)
# Refer to matplotlib doc for plt functions: https://matplotlib.org/stable/tutorials/pyplot.html

plt.plot(i, t, 'r*-', label='Target') # red star marker solid line (r*-)
plt.plot(i,x[0],'bo-',label='Input 1') # blue circle marker solid line (bo-)
plt.plot(i,x[1],'bs-',label='Input 2') # blue square marker solid line (bs-)
plt.title('Training data')
plt.legend()
plt.show()

# forward propagation
for k in range(s):
  x0_k = x[0, k]   # kth data point of 1st input
  x1_k = x[1, k]   # kth data point of 2nd input
  zini[k] = (wi1 * x0_k + wi2 * x1_k) + bi
  zi[k] = (2/(1+np.exp(-2*zini[k])))-1 # activation function
  yi[k] = zi[k] * wo + bo # network output


# Plotting network output without training
plt.plot(i,t,'r*-',label='Target')
plt.plot(i,yi,'k+-',label='Output')
plt.title('Network output without training')
plt.legend()
plt.show()