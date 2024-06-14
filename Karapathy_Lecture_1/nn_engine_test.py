from nn_engine import Neuron,MLP, Layer

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

#initialize NN MLP layer

n = MLP(3,[4,4,1])

for k in range(100):
    #forward pass
    ypred = [n(i) for i in xs]
    loss = sum([(ypd - yct)**2 for ypd,yct in zip(ypred,ys)])

    #backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward

    #update grad
    for p in n.parameters():
        p.data += -0.01 * p.grad 

    print(f'step :{k} & loss : {loss}')