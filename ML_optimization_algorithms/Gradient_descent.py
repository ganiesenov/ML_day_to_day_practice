import numpy as np 

def generate_data(): 
    np.random.seed(42) 
    x = np.random.randn(100,1 )
    y = 5 * x + np.random.randn(100,1)
    return x, y 

def create_model(x, y): 
    weight = 0.0 
    bias = 0.0 
    learning_rate = 0.01 
    epochs = 100 
    loss = []
    
    for epoch in range(epochs): 
        grad_w = 0.0 
        grad_b = 0.0 
        N = x.shape[0]
        
        for i in range(N): 
            x_i = x[i][0]
            y_i = y[i][0] 
            # Loss = 1/N * ((y - y_pred)**2)  = U**2
            # Chain Rule u = (y_i - weight*x_i -bias)
            # U**2 = 2u * u' = 2u * (y_i - weight *x_i - bias) * (-x_i)
            #  U **2 = - 2u *x_i +bias  
            
            
            y_hat = weight * x_i + bias 
            error = y_i - y_hat 
            
            grad_w += -2 * x_i * error 
            grad_b += -2 *error 
        
        weight -= grad_w * learning_rate/ N 
        bias -= grad_b * learning_rate / N 
        
        y_pred = weight * x + bias 
        
        loss.append(np.mean((y-y_pred)**2)) 
        
        if epoch % 10 == 0: 
            print (f"Loss : {loss[-1]:.4f}, Epochs : {epoch}, Weight : {weight:.2f}, Bias : {bias:.2f}")
            
    return  weight, bias, loss 
    
x,y = generate_data() 
weight, bias, loss = create_model(x,y)