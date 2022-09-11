# -*- coding: utf-8 -*-
"""

@author: Alex Vong B827861

Developed using spyder IDE
some guidance from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""


'''imports'''
import math
import random
#reads from excel
import pandas as pd

'''initialise the network'''
def initialise(inputn,hiddenn,outputn): #takes number of inputs, hidden nodes (assuming 1 hidden layer) and outputs as parameters
    network = list() #nested list of weights/biases for each node in the network
    #generate weights and bias for each node in the hidden layer
    hiddenlayer = [{"weights":[random.random() for i in range(inputn+1)]}for j in range(hiddenn)] #{"n"}: splits the list into sections
    network.append(hiddenlayer)
    #generate weights and bias for output layer
    outputlayer = [{"weights":[random.random() for i in range(hiddenn+1)]}]
    network.append(outputlayer)
    return network
    #print(network)

'''calculates weighted sum'''
def weightedsum(weights, inputs):
    wsum = weights[-1] #assume bias to be last value in list
    for i in range(len(weights)-1): #for each weight (-1 to exclude bias)
        wsum += weights[i] * inputs[i] #sum of weight x input
    return wsum

'''activation/transfer function'''
def activationf(wsum):
    activation = 1 / (1 + math.exp(-wsum))
    return activation


'''derivative of the transfer function'''
def transferderiv(output):
    return output * (1.0 - output)

'''
Backpropagation algorithm
#########################################################################################################################
'''    

'''does the forward propagation'''
def forwardprop(network, inputv): #takes the network and input values as parameters
    inputs = inputv
    for layer in network: #for each layer in the network
        newinputs =[] #inputs for next layer
        for node in layer: #for each node in the layer
            s = weightedsum(node["weights"],inputs) #find weighted sum at node
            node["output"] = activationf(s) #return transfer function of node
            newinputs.append(node["output"])
        inputs = newinputs
    return inputs #returns the outputs for each node


'''does the backpropagation'''
def backprop(network,expected):
    for i in reversed(range(len(network))): #work backwards through the network
        layer = network[i] #current layer
        #if not at output layer (at hidden layer)
        if i != len(network)-1:
            for j in range(len(layer)):
                nodeh = layer[j] #get node at layer
                #get weight and delta of output layer
                for node in network[i+1]: #for loop seems to be the only way to get the values
                    outweight = node['weights'][j]
                    outdelta = node["delta"]
                #calculate delta at node (weight of node to output x delta at output x (output * (1 - output)))
                nodeh["delta"] = (outweight* outdelta * transferderiv(node["output"]))
     
        #at output layer        
        else:
            #find delta of output node
            node = layer[0] #only 1 output node so will be the first node in the layer
            #get error (expected output - actual output)
            errorout = expected - node["output"]
            outdelta = errorout * transferderiv(node["output"]) #get delta
            node["delta"] = outdelta #set delta for node
  
       
'''update weight of nodes'''
def updateweights(network, data, lr):
    for i in range (len(network)): #for each layer in the network
        inputs = data[:-1] #all data except for the last
        if i !=0: #if the output layer
            inputs = [node["output"]for node in network[i - 1]] #inputs are the outputs of prevous layer
            
        for node in network[i]: #for each node in the layer
            for j in range(len(inputs)): #for each weight
                node['weights'][j] += lr * node['delta'] * inputs[j] #add learning rate x delta x output
            node['weights'][-1] += lr * node['delta'] #for bias output is always 1  

                

    

'''
#########################################################################################################################
'''    


'''train the network'''
def trainnetwork(network, data, lr, epochs, outputs,):
    for epoch in range(epochs): #iterate through epochs
            sumerror = 0
            for row in data: #for each data entry
                outputs = forwardprop(network,row) #forward pass
                expected = row[-1]
                sumerror += (expected-outputs[0])**2 #sum of the errors squared
                backprop(network,expected) #backwards pass
                updateweights(network,row,lr) #update the weights
            rmse = math.sqrt(sumerror/len(data)) #calculate root mean squared error for epoch
            #print('>epoch=%d, lrate=%.3f, RMSE=%.3f' % (epoch+1, lr, rmse)) #print the epoch number, learning rate and RMSE for that epoch
    print("RMSE=%.3f" % (rmse))



'''
Improvements
#########################################################################################################################
''' 


'''updates weights with momentum added'''          
def updatewithmomentum(network, data, lr, mfunction):
    for i in range (len(network)): #for each layer in the network
        inputs = data[:-1] #all data except for the last
        if i !=0: #if the output layer
            inputs = [node["output"]for node in network[i - 1]] #inputs are the outputs of prevous layer
            
        for node in network[i]: #for each node in the layer
            for j in range(len(inputs)): #for each weight
                wchange =lr * node['delta'] * inputs[j] #calculate change in weight
                node['weights'][j] += wchange + (mfunction * wchange)  #add weight + momentum
                wchange = lr * node['delta']#for bias output is always 1
                node['weights'][-1] += wchange + (mfunction * wchange) 
                
            
'''train the network with momentum'''
def trainwithmomentum(network, data, lr, epochs, outputs,):
    for epoch in range(epochs): #iterate through epochs
            sumerror = 0
            for row in data: #for each data entry
                outputs = forwardprop(network,row) #forward pass
                expected = row[-1]
                sumerror += (expected-outputs[0])**2
                backprop(network,expected) #backwards pass
                updatewithmomentum(network,row,lr,0.9) #update the weights
            rmse = math.sqrt(sumerror/len(data)) #calculate root mean squared error for epoch
            #print('>epoch=%d, lrate=%.3f, RMSE=%.3f' % (epoch+1, lr, rmse)) #print the epoch number, learning rate and RMSE for that epoch
    print("RMSE=%.3f" % (rmse))

'''train the network with bold driver'''
def trainwithbdriver(network, data, lr, epochs, outputs,):
    rmse = 0
    learning_rate = lr
    for epoch in range(epochs): #iterate through epochs
        errorincrease = True
        preverror = rmse
        networkcopy = network #make a backup of the network
        if (epoch+1) % 1000 == 0: #every 1000 epochs try bold driver
            print("implementing bold driver")
            while errorincrease == True:
                sumerror = 0
                for row in data: #for each data entry
                    outputs = forwardprop(network,row) #forward pass
                    expected = row[-1]
                    sumerror += (expected-outputs[0])**2
                    backprop(network,expected) #backwards pass
                    updateweights(network,row,learning_rate) #update the weights
                rmse = math.sqrt(sumerror/len(data)) #calculate root mean squared error for epoch
                if rmse - preverror > 0: #if the error function increased and it isnt the first epoch
                    print("error increased")
                    network = networkcopy #restore network to previous iteration
                    if learning_rate * 0.5 >= 0.01: #keep within the boundary
                        print("learning rate decreased")
                        learning_rate = learning_rate *0.5 #half the learning rate
                    else:
                        print("learning rate unchanged")
                        errorincrease = False
                else:
                    print("error decreased")
                    if learning_rate * 1.1 <= 0.5 :
                        print("learning rate increased")
                        learning_rate = learning_rate *1.1
                    errorincrease = False 

                print("learning rate: "+str(learning_rate))
        else:
            sumerror = 0
            for row in data: #for each data entry
                outputs = forwardprop(network,row) #forward pass
                expected = row[2]
                sumerror += (expected-outputs[0])**2
                backprop(network,expected) #backwards pass
                updateweights(network,row,learning_rate) #update the weights
            rmse = math.sqrt(sumerror/len(data)) #calculate root mean squared error for epoch
        #print('>epoch=%d, lrate=%.3f, RMSE=%.3f' % (epoch+1, lr, rmse)) #print the epoch number, learning rate and RMSE for that epoch
    print("RMSE=%.3f" % (rmse))

'''calculate learning rate through annealing'''
def calculatelr(epochn,maxepoch, startlr, endlr):
    lr = endlr + ((startlr - endlr)*(1 - (1/ (1+ math.exp(10 - ((20 * epochn)/maxepoch))))))
    return lr
    
'''train the network with annealing'''
def trainwithanneal(network, data, lr, epochs, outputs):
    for epoch in range(epochs): #iterate through epochs
        learning_rate = calculatelr(epoch, epochs, 0.1, 0.01) #calculate leraning rate for this epoch
        sumerror = 0
        for row in data: #for each data entry
            outputs = forwardprop(network,row) #forward pass
            expected = row[-1]
            sumerror += (expected-outputs[0])**2 #sum of the errors squared
            backprop(network,expected) #backwards pass
            updateweights(network,row,learning_rate) #update the weights
        rmse = math.sqrt(sumerror/len(data)) #calculate root mean squared error for epoch
        #print('>epoch=%d, lrate=%.3f, RMSE=%.3f' % (epoch+1, lr, rmse)) #print the epoch number, learning rate and RMSE for that epoch
    print("RMSE=%.3f" % (rmse))
    

'''
#########################################################################################################################
'''       

'''validate the network'''
def validatenetwork(network, data, lr, epochs, outputs):
    preverror = 0
    rmse = 10 # set the initial rmse to a high value so that it doesnt stop early
    stop = False
    for epoch in range(epochs): #iterate through epochs
        sumerror = 0
        preverror = rmse #get error of previous epoch
        if stop == False: #as long as error isnt increasing
            for row in data: #for each data entry
                outputs = forwardprop(network,row) #forward pass
                expected = row[-1]
                sumerror += (expected-outputs[0])**2
                backprop(network,expected) #backwards pass
                updateweights(network,row,lr) #update the weights
            rmse = math.sqrt(sumerror/len(data)) #calculate root mean squared error for epoch
            #print('>epoch=%d, lrate=%.3f, RMSE=%.3f' % (epoch+1, lr, rmse)) #print the epoch number, learning rate and RMSE for that epoch
            if preverror - rmse < 0: #if error increases
                print("STOPPED EARLY") #notify if stopped early
                stop = True
                break
        else:
            break
    print("RMSE=%.3f" % (rmse))
    return epoch

'''calculate outputs using the network and find average error'''
'''doesn't backpropagate'''
def testnetwork(network, data):
    sumerror = 0
    for row in data: #for each data entry
        outputs = forwardprop(network,row) #forward pass
        expected = row[-1]
        sumerror += (expected-outputs[0])**2 #sum of the errors squared
    rmse = math.sqrt(sumerror/len(data)) #calculate root mean squared error for epoch
    print('RMSE=%.3f' % (rmse)) #print the epoch number, learning rate and RMSE for that epoch
        
    

'''create training dataset from excel'''
def createtrainingdata():
    #read excel spreadsheet
    print("function started")
    df = pd.read_excel('TestData.xlsx',sheet_name="Training")
    print("Spreadsheet opened")
    #get inputs and output from fields in spreadsheet
    input1 = df['sT'].tolist() #input 1
    input2 = df['sW'].tolist() #input 2
    input3 = df['sSr'].tolist() #input 3
    input4 = df['sDSP'].tolist() #input 4
    input5 = df['sDRH'].tolist() #input 5
    output = df['sPanE'].tolist() #predictand
    print("Pulled data")
    dataset = [[input1[i],input2[i],input3[i],input4[i],input5[i],output[i]] for i in range(len(input1))] #form data entries as nested list
    print("data formed into list")
    return dataset

'''create validation dataset from excel'''
def createvalidationdata():
    #read excel spreadsheet
    print("function started")
    df = pd.read_excel('TestData.xlsx',sheet_name="Validation")
    print("Spreadsheet opened")
    #get inputs and output from fields in spreadsheet
    input1 = df['sT'].tolist() #input 1
    input2 = df['sW'].tolist() #input 2
    input3 = df['sSr'].tolist() #input 3
    input4 = df['sDSP'].tolist() #input 4
    input5 = df['sDRH'].tolist() #input 5
    output = df['sPanE'].tolist() #predictand
    print("Pulled data")
    dataset = [[input1[i],input2[i],input3[i],input4[i],input5[i],output[i]] for i in range(len(input1))] #form data entries as nested list
    print("data formed into list")
    return dataset


'''create test dataset from excel'''
def createtestdata():
    #read excel spreadsheet
    print("function started")
    df = pd.read_excel('TestData.xlsx',sheet_name="Testing")
    print("Spreadsheet opened")
    #get inputs and output from fields in spreadsheet
    input1 = df['sT'].tolist() #input 1
    input2 = df['sW'].tolist() #input 2
    input3 = df['sSr'].tolist() #input 3
    input4 = df['sDSP'].tolist() #input 4
    input5 = df['sDRH'].tolist() #input 5
    output = df['sPanE'].tolist() #predictand
    print("Pulled data")
    dataset = [[input1[i],input2[i],input3[i],input4[i],input5[i],output[i]] for i in range(len(input1))] #form data entries as nested list
    print("data formed into list")
    return dataset

'''test code'''
def test(): 
    print("PROGRAM START")
    trainingdata = createtrainingdata()
    validationdata = createvalidationdata()
    testdata = createtestdata()
    #sample data for quick testing
    sampledata = [[0.32,0.36,0.19,0.68,0.71,0.14],[0.38,0.64,0.22,0.35,0.69,0.18],[0.36,0.51,0.19,0.51,0.72,0.17]]
    sampledata2 = [[0.3,0.7,0],[0.7,0.3,0],[0.7,0.2,1.0],[0.8,0.3,1.0],[0.6,0.3,1.0],[0.7,0.4,1.0],[0.3,0.6,1.0]]
    #initialise network
    print("DATA PROCESSED")
    #set number of inputs and hidden nodes here
    network = initialise(5, 5, 1)
    networkcopy1 = network
    linestnetwork = [[{"weights":[0.53864,0.247723,0.393995,0.033318,-0.15136,-0.16677]}]]

    print("NETWORK INITIALISED")
    #train network
    trainwithanneal(network, trainingdata, 0.1, 2000, 1)
    print("NETWORK TRAINED")
    #validate network
    validatenetwork(networkcopy1, validationdata, 0.1, 2000, 1)

    #test the network
    testnetwork(network, testdata)
    testnetwork(linestnetwork,testdata)
    print("NETWORK TESTED")
    #end of program
    print("PROGRAM COMPLETE")

test()
 
            
            
            
    
    
            