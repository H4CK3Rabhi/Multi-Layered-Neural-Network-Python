import numpy as np
def matmul(a,q):
    x=a.tolist()
    y=q.tolist()
    qw=np.matrix(x)
    qe=np.matrix(y)
    mul=qw*qe
    l=mul.tolist()
    return np.array(l)

class NeuralNetwork:

    def __init__(self,numi,numh,numo):#numi- no of input layers # numh=no of hidden neurons
          # numo-no of outputs layer
          self.input_nodes=numi
          self.hidden_nodes=numh
          self.output_nodes=numo
          self.weights_ih=np.empty([self.hidden_nodes,self.input_nodes])
          self.weights_ho=np.empty([self.output_nodes,self.hidden_nodes])
          self.bias_h=np.empty([self.hidden_nodes])
          self.bias_o=np.empty([self.output_nodes])
          self.lr=0.0997
          for i in range(self.hidden_nodes):
              self.bias_h[i]=np.random.randn()
              for j in range(self.input_nodes):
                  self.weights_ih[i][j]=np.random.randn()
          for i in range(self.output_nodes):
              self.bias_o[i]=np.random.randn()
              for j in range(self.hidden_nodes):
                  self.weights_ho[i][j]=np.random.randn()

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoidp(self,x):
        return sigmoid(x)*(1-sigmoid(x))

    def feedforward(self,inputq=[]):
        inputs=np.array(inputq,np.dtype('float64'))
        h=np.add(np.dot(self.weights_ih,inputs),self.bias_h)
        hidden=np.empty([len(h)])
        for i in range(len(h)):
            hidden[i]=self.sigmoid(h[i])
        
        output=np.add(np.dot(self.weights_ho,hidden),self.bias_o)
        guess=np.empty(len(output))
        for i in range(len(output)):
            guess[i]=self.sigmoid(output[i])   
        return guess.tolist()

    def train(self,inputq=[],answers=[]):
        targets=np.array(answers,np.dtype('float64'))
        inputs=np.array(inputq,np.dtype('float64'))
        h=np.add(np.dot(self.weights_ih,inputs),self.bias_h)
        hidden=np.empty([len(h)])
        for i in range(len(h)):
            hidden[i]=self.sigmoid(h[i])
        output=np.add(np.dot(self.weights_ho,hidden),self.bias_o)
        outputs=np.empty(len(output))
        for i in range(len(output)):
            outputs[i]=self.sigmoid(output[i])
        output_error=np.subtract(targets,outputs)
        who_t=self.weights_ho.T
        hidden_error=np.dot(who_t,output_error)
        VWho=np.empty([self.output_nodes])
        VWih=np.empty([self.hidden_nodes])
        for i in range(self.output_nodes):
                VWho[i]=self.lr*output_error[i]*(outputs[i]*(1-outputs[i]))
        for i in range(self.hidden_nodes):
                VWih[i]=self.lr*hidden_error[i]*(hidden[i]*(1-hidden[i]))
        tin=np.empty([self.input_nodes,1])
        th=np.empty([self.hidden_nodes,1])
        for i in range(self.input_nodes):
            for j in range(1):
                tin[i][j]=inputs[i]
        for i in range(self.hidden_nodes):
            for j in range(1):
                th[i][j]=hidden[i]
        VWho_f=matmul(th,VWho)
        VWih_f=matmul(tin,VWih)
        self.weights_ih=np.add(self.weights_ih,VWih_f.T)
        self.weights_ho=np.add(self.weights_ho,VWho_f.T)
        self.bias_o=np.add(self.bias_o,VWho)
        self.bias_h=np.add(self.bias_h,VWih)
if __name__=="__main__" :
    nn=NeuralNetwork(2,4,1)
    inputq=[[0,0],[0,1],[1,0],[1,1]]
    target=[[0],[1],[1],[0]]
    for i in range(10600): 
        ri=np.random.randint(len(inputq))
        nn.train(inputq[ri],target[ri])
    output=nn.feedforward([0,1])
    print(output)
