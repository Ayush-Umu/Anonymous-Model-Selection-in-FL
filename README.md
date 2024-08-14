# Anonymous-Model-Selection-in-FL
This is a repository for the paper 'Anonymous Model Selection in Federated Learning with Convergence Guarantee' under review. The preprint can be found at: [10.36227/techrxiv.170327604.45388443/v1](https://www.techrxiv.org/doi/full/10.36227/techrxiv.170327604.45388443)

The flowchart of the algorithm is given below:
<img width="600" alt="Screenshot 2024-08-14 at 13 57 38" src="https://github.com/user-attachments/assets/bc968406-7c19-4af9-8d01-1d185f3d9a48">

We have considered the iid and non-iid variation of MNIST (60K train and 10K test images), FashionMNIST (60K train and 10K test images) and CIFAR10 (50K train and 10K test images) datasets. The experimental setup is given below:

<img width="600" alt="Screenshot 2024-08-14 at 14 01 30" src="https://github.com/user-attachments/assets/e712cafe-a201-4cf2-8b62-197d2049e233">

Repos:
DP_fedavg: contain the code for differentially private fedavg and fedProx for iid and non-iid MNIST. 
Fedavg models: contain the code for fedavg and fedProx for iid and non-iid MNIST.
IP_fedavg: contain the code for integrally private fedavg and fedProx for iid and non-iidMNIST.
device_records.p: In each repository, this file gives the number of records on each device for non-iid data.
