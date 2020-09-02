import numpy as np
np.random.seed(22)

def load_training_data():

    training_data = np.loadtxt("train.txt")
    dim = training_data.shape[1];
    

    train_X = training_data[:,:-1]
    train_Y = training_data[:, -1]

    
    


    y = np.array(train_Y)
    x = np.array(train_X)

        

    b = np.ones((len(y), dim))

    

    b[:, :-1] = x
    x = b

    

    return x,y,len(y)


def initialize_2D(classes,samples):
    w, h = classes, samples
    result = [[0 for x in range(w)] for y in range(h)]
    return result

def convertToOneHot(y,classes,samples):
    result = np.zeros(shape=(classes,samples))
    for i in range(0,samples):
        ind = y[i]
        result[(int)(float(ind))-1][i] = 1
    result = np.transpose(result)
    return result

def initialize_weights(network):
    np.random.seed(2)
    weights = {}  # dictionary indexed by w1, w2, w3...
    for i in range(1, len(network)):
        weights['w' + str(i)] = np.random.randn(network[i], network[i - 1])


    return weights

def sigmoid(x):

    return 1/(1+np.exp(-x))


def forward_pass(x,weights,layers):
    summations = {}  # dictionary indexed by v1, v2, v3.
    outputs = {}  # dictionary indexed by y0, y1, y2.

    y = np.transpose(x)
    outputs['y' + str(0)] = y


    for i in range (1,layers):

        y_prev = y
        v = np.dot(weights["w"+str(i)],y_prev)
        summations['v' + str(i)] = v

        y = v+1-1

        for k in np.nditer(y, op_flags=['readwrite']):  # Using sigmoid function
            k[...] = sigmoid(k)
        outputs['y' + str(i)] = y


    return summations,outputs


def cost_function(actual_y,predicted_y,classes):

    sample_cost = 0
    last_layer_errors = initialize_2D(1,classes)  # errors of the last layer,for each neuron in case of each sample

    for i in range (0,classes):
        last_layer_errors[i] = actual_y[i]- predicted_y[i]
        sample_cost += pow(actual_y[i]- predicted_y[i],2)

    sample_cost = 0.5*sample_cost
    last_layer_errors = np.array(last_layer_errors)
    return sample_cost,last_layer_errors

def calculate_del_L(last_layer_errors,summations,classes,layers):
    f_prime = summations['v' + str(layers - 1)]

    for x in np.nditer(f_prime, op_flags=['readwrite']):
        x[...] = sigmoid(x) * (1 - sigmoid(x))
    f_prime = np.transpose(f_prime)

    del_L = initialize_2D(1,classes)

    for i in range(0, classes):
        del_L[i] = last_layer_errors[i] * f_prime[i]
    del_L = np.array(del_L)
    return del_L


def backward_pass(del_L,weights,summations,layers):

    dels = {}
    ers = {}

    dels['d' + str(layers - 1)] = del_L  # last layer
    for i in range(layers - 1, 1, -1):
        ers['e' + str(i - 1)] = np.dot(dels['d' + str(i)], weights['w' + str(i)])  # e(r-1) = del(r) * w(r)

        f_prime = summations['v' + str(i - 1)]
        for x in np.nditer(f_prime, op_flags=['readwrite']):
            x[...] = sigmoid(x) * (1 - sigmoid(x))
        f_prime = np.transpose(f_prime)
        temp_del = initialize_2D(1, f_prime.shape[0])

        """for p in range(0, f_prime.shape[0]):
            for q in range(0, f_prime.shape[1]):
                temp_del[p][q] = ers['e' + str(i - 1)][p][q] * f_prime[p][q]  # del(r-1) = e(r-1) = f_prime(r-1)
        """
        for p in range(0,f_prime.shape[0]):
            temp_del[p] = ers['e' + str(i - 1)][p] * f_prime[p]  # del(r-1) = e(r-1)*f_prime(r-1)
        temp_del = np.array(temp_del)
        dels['d' + str(i - 1)] = temp_del

    return dels



def test_inputs():
    y = np.loadtxt("test.txt")
    dim = y.shape[1];
    x = y[:, :-1]
    y = y[:, dim - 1]
    y = np.array(y)
    x = np.array(x)

    b = np.ones((len(y), dim))

    b[:, :-1] = x
    x = b
    return x, y, len(y)

def derivative(output):
	return output * (1.0 - output)

def backward_error(network,weights,outputs,last_layer_errors):
    neuron = {}
    for i in reversed(range(len(network))):
        errors = list()
        if i != len(network) - 1:
            for j in range(network[i]):
                error = 0.0
                for n in range(network[i+1]):
                    error+= weights['w'+str(i+1)][n][j] * neuron['delta'+str(i+1)+str(n)]
                errors.append(error)

        else:
            for j in range(network[i]):
                errors.append(last_layer_errors[j])
        for j in range(network[i]):
                neuron['delta'+str(i)+str(j)] = errors[j]* derivative( outputs['y'+str(i)][j])


    return neuron


def update_weights(network,deltas,row,outputs,weights,learning_rate):
    for i in range(1,len(network)):
        inputs = row
        if(i!=1):
            inputs = outputs['y'+str(i-1)]
        for p in range(network[i]):
            for q in range(len(inputs)):
                weights['w'+str(i)][p][q] += learning_rate*deltas['delta'+str(i)+str(p)]*inputs[q]

    return weights


def main():
    # network = [3, 10,5, 3];
    #learning_rate = 0.6

    row,out,samples = load_training_data();
    total_classes = []
    total_classes = set(total_classes)
    for o in out:
        total_classes.add(o)
    classes = len(total_classes)



    attributes = row.shape[1]
    hid_layers =  int(input("How many hidden layers? "))
    network = []
    network.append(attributes)
    for i in range(hid_layers):
        a = int(input("dimension of layer? "))
        network.append(a)


    learning_rate = float(input("Learning rate? "))


    network.append(classes)
    #print(network)
    layers = len(network)
    out = convertToOneHot(out, classes, samples)
    weights = initialize_weights(network)


    for m in range(0,50):

        total_cost = 0

        for i in range(0,samples):
            x = row[i]
            y = out[i]

            summations, outputs = forward_pass(x, weights, layers)

            sample_cost,last_layer_errors = cost_function(y, np.transpose(outputs['y' + str(layers - 1)]), classes)
            deltas = backward_error(network, weights, outputs, last_layer_errors)

            weights =update_weights(network,deltas,x,outputs,weights,learning_rate)
            total_cost+=sample_cost

        print("training loss " +str(total_cost))

    x,y,samples = test_inputs()
    summations, outputs = forward_pass(x, weights, layers)
    predictions = np.transpose(outputs['y' + str(layers - 1)])

    correct = 0
    for i in range (0,samples):
        max = -100.0
        m_index = -1
        for j in range (0,classes):
            if(predictions[i][j]>max):
                max = predictions[i][j]
                m_index = j+1
        if(m_index==y[i]):
           correct = correct+1

    #print(correct)
    accuratcy = correct/samples
    print("test accuracy " + str(accuratcy*100))


if __name__ == '__main__':
    main()
