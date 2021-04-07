import pickle
import matplotlib.pyplot as plt
import seaborn as sns

f = open('./learning_curves/exp_torch_4/rewards.pkl','rb')
f1 = open('./learning_curves/exp_torch_5/rewards.pkl','rb')
f2 = open('./learning_curves/exp_torch_8/rewards.pkl','rb')

rewards = pickle.load(f)
rewards1 = pickle.load(f1)
rewards2 = pickle.load(f2)
plt.plot(rewards)
plt.plot(rewards1)
plt.plot(rewards2)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title('Rewards')
plt.legend(['4', '5', '6'])
plt.show()
