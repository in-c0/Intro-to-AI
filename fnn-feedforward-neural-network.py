import numpy as np
import matplotlib.pyplot as plt

# Define input and target vectors
x = np.array([[-2, -1, 0, 1],  # input 1
              [-1,  0, 1, 2]]) # input 2

t = np.array([-1.5, -1, 1, 1.5]) # target

# Forward propagation variables
s = 4         # number of samples
zini = np.zeros(s) # zin for each data point
zi = np.zeros(s)   # zi for each data point
yi = np.zeros(s)   # output without training

# Weights and biases
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


#Training algorithm

mse=np.zeros(q)
e=np.zeros(q)

for ep in range(q):
    dEdbo=dEdwo=dEdbi=dEdwi1=dEdwi2=0
    zin=np.zeros(s)
    z=np.zeros(s)
    y=np.zeros(s)
    for k in range(s):
        zin[k]=wi1*x[0,k]+wi2*x[1,k]+bi
        z[k]= (2.0/(1+np.exp(-2*zin[k])))-1
        y[k]=wo*z[k]+bo #output of the network
        e[k]=y[k]-t[k] #computing the error
        mse[ep]=mse[ep]+(1.0/s)*np.power(e[k],2) #computing the mean squared error

        dEdbo=dEdbo+a*(2.0/s)*e[k] # delta E with respect to output bias ... Z(t) = 1 is ommitted for output bias
        dEdwo=dEdwo+a*(2.0/s)*e[k]*z[k] # delta E with respect to output weight ... Z(t) is z[k]
        dEdbi=dEdbi+a*(2.0/s)*e[k]*wo*(4*np.exp(-2*zin[k])/np.power(1+np.exp(-2*zin[k]),2))
        dEdwi1=dEdwi1+a*(2.0/s)*e[k]*wo*(4*np.exp(-2*zin[k])/(np.power(1+np.exp(-2*zin[k]),2)))*x[0,k]
        dEdwi2=dEdwi2+a*(2.0/s)*e[k]*wo*(4*np.exp(-2*zin[k])/(np.power(1+np.exp(-2*zin[k]),2)))*x[1,k]

    #Updating network parameters
    wi1=wi1-dEdwi1
    wi2=wi2-dEdwi2
    bi=bi-dEdbi
    wo=wo-dEdwo
    bo=bo-dEdbo

# Plotting the mean squared error
plt.semilogy(range(q),mse,'b.', label='MSE')
plt.title('Mean squeared error')
plt.xlabel('epochs')
plt.ylabel('performance')
plt.legend()
plt.show()


# Recalculate the network output after training with updated weights and biases
for k in range(s):
    x0_k = x[0, k]   # kth data point of 1st input
    x1_k = x[1, k]   # kth data point of 2nd input
    zini[k] = (wi1 * x0_k + wi2 * x1_k) + bi
    zi[k] = (2/(1+np.exp(-2*zini[k])))-1 # activation function
    yi[k] = zi[k] * wo + bo # updated network output

# Plotting network output after training
plt.plot(i,t,'r*-',label='Target')
plt.plot(i,yi,'k+-',label='Output after training')  # Now plotting the updated output
plt.title('Network output after training')
plt.legend()
plt.show()
