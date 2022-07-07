import numpy as np
import matplotlib.pyplot as plt

# This code produces the loss/training plot for Figure 3 b).

path = 'D:/Code/QNN/effective_dimension/Loss_plots/'

stddevs =[]
averages =[]
# colors:
rooi = np.array([255, 29, 0])/255
blou = np.array([0, 150, 236])/255
groen = np.array([0,208,0])/255

# Load classical data
loss = np.zeros((100,100))
for i in range(100):
    file = path + 'data/classical/classical_loss_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)

sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
# print("classical loss: ", av) 
plt.plot(range(100), av, label='classical neural network', color=rooi)
# plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=rooi)
stddevs.append(sd)
averages.append(av)

# Load easy qnn data
loss_eqnn_d1 = np.load(path+'data/easy_qnn/quantum_loss_easy_99.npy')
loss_eqnn_d1 = np.reshape(np.array(loss_eqnn_d1), (100, 100))
sd = np.std(loss_eqnn_d1, axis=0)
av = np.average(loss_eqnn_d1, axis=0)
# print("easy qnn loss: ", av)
# plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=blou)
plt.plot(range(100), av, label='easy quantum model', color=blou)
stddevs.append(sd)
averages.append(av)


# full quantum neural network
loss = np.zeros((100,100))
for i in range(100):
    file = path+'data/hard_qnn/quantum_loss_hard_dep2_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)

sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
# print("hard qnn loss: ", av)
plt.plot(range(100), av, label='full quantum neural network', color=groen)
# plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=groen)
stddevs.append(sd)
averages.append(av)

# custom ZZ quantum neural network
loss = np.zeros((36,100))
for i in range(36):
    file = 'D:/Code/QNN/effective_dimension/Loss_plots/data-custom/' + 'quantum_loss_hard_dep2_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)
#print(loss[0])
sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
# print("hard-custom qnn loss: ", av)
plt.plot(range(100), av, label='custom ZZ quantum neural network', color='yellow')
# plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color='yellow')
stddevs.append(sd)
averages.append(av)

# custom Z quantum neural network
loss = np.zeros((1,100))
for i in range(1):
    file = 'D:/Code/QNN/effective_dimension/Loss_plots/data-custom/' + 'quantum_loss_hard_circle_dep2_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)
#print(loss[0])
sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
# print("hard-custom qnn loss: ", av)
plt.plot(range(100), av, label='custom Z quantum neural network', color='orange')
# plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color='yellow')
stddevs.append(sd)
averages.append(av)

# linear quantum neural network
loss = np.zeros((2,100))
for i in range(2):
    file = 'D:/Code/QNN/effective_dimension/Loss_plots/data-custom/' + 'quantum_loss_hard_dep2_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)
#print(loss[0])
sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
# print("hard-custom qnn loss: ", av)
# plt.plot(range(100), av, label=' linear quantum neural network', color='violet')
# plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color='yellow')
stddevs.append(sd)
averages.append(av)


# GradientDes quantum neural network
loss = np.zeros((19,100))
for i in range(19):
    file = 'D:/Code/QNN/effective_dimension/' + 'quantum_loss_hard_full_grades_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)
# for i in range(17):
#     file = 'D:/Code/QNN/effective_dimension/' + 'quantum_loss_hard_full_grades_%d.npy'%i
#     print(np.load(file, allow_pickle=True))
sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
# print("hard-custom qnn loss: ", av)
plt.plot(range(100), av, label=' GradientDes quantum neural network', color='violet')
# plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color='yellow')
stddevs.append(sd)
averages.append(av)
# IBMQ Montreal raw data
loss_ibmq_montreal = [
    0.5864, 0.5115, 0.4597, 0.4062, 0.3654, 0.3390, 0.3330, 0.3339, 0.3241, 0.3276, # 10
    0.3234, 0.3038, 0.2978, 0.2728, 0.2598, 0.2575, 0.2486, 0.2564, 0.2653, 0.2712, # 20
    0.2668, 0.2809, 0.2638, 0.2652, 0.2551, 0.2453, 0.2386, 0.2543, 0.2440, 0.2404, # 30
    0.2417, 0.2278, 0.2235]

loss_ibmq_montreal_with_stable = [
    0.5864, 0.5115, 0.4597, 0.4062, 0.3654, 0.3390, 0.3330, 0.3339, 0.3241, 0.3276, # 10
    0.3234, 0.3038, 0.2978, 0.2728, 0.2598, 0.2575, 0.2486, 0.2564, 0.2653, 0.2712, # 20
    0.2668, 0.2809, 0.2638, 0.2652, 0.2551, 0.2453, 0.2386, 0.2543, 0.2440, 0.2404, # 30
    0.2417, 0.2278, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235]

averages.append(loss_ibmq_montreal_with_stable)

plt.plot(loss_ibmq_montreal, label='ibmq_montreal backend', color='black')
plt.plot(loss_ibmq_montreal_with_stable, '--', color='black')
plt.ylabel('loss value')
plt.xlabel('number of training iterations')
plt.legend()
plt.savefig('loss_with_std_dev.pdf', format='pdf', dpi=1000)
plt.show()

# save source data as text files
np.savetxt('average_loss_values.txt', averages)
np.savetxt('std_dev_of_loss.txt', stddevs)