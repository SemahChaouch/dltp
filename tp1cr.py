import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def acti_func(z):
    if z>0:
        return 0
    else :
        return -1
    
mu1 = np.array([-1, 0])
mu2 = np.array([1, 0])
sigma1 = 0.25
sigma2 = 0.25

X1 = np.random.normal(mu1, sigma1, size=(125, 2))
y1 = -np.ones(125)
X2 = np.random.normal(mu2, sigma2, size=(125, 2))
y2 = np.zeros(125)
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)

np.random.seed(42)
shuffle_idx = np.random.permutation(len(X))
X_shuffled = X[shuffle_idx]
y_shuffled = y[shuffle_idx]

X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)


    
def perceptron(X, y, lr, epochs):
    m, n = X.shape
    w = np.zeros((n+1,1))
    for epoch in range(epochs):
        for idx, x_i in enumerate(X):
            x_i = np.insert(x_i, 0, 1).reshape(-1,1)
            # Calculating prediction/hypothesis.
            y_hat = acti_func(np.dot(x_i.T, w))
            if (np.squeeze(y_hat) - y[idx]) != 0:
                w += lr*((y[idx] - y_hat)*x_i)
    return w

w= perceptron(X_train,y_train,0.1,100)
y_pred=[]
for idx,xi in enumerate(X_test):
    xi = np.insert(xi, 0, 1).reshape(-1,1)
    y_pred.append(acti_func(np.dot(xi.T, w)))

# Calculer la précision
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
# Tracer les points d'entraînement
plt.ylim([-0.6, 0.6])
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color="blue")
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color="red")
#plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color="blue")
#plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color="red")
x_points=np.array([min(X[:,0]), max(X[:,0])])
m = -w[1]/w[2]
c = -w[0]/w[1]
y_points = m*x_points + c
plt.plot(x_points, y_points, color="black")
plt.legend(["Classe 0", "Classe -1", "Limite"])
plt.title("Droite de limite de décision pour le Perceptron")
plt.show()

