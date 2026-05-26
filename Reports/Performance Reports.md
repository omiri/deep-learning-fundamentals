1. Problem Statement
Manual identification of plant taxonomy species based on physical characteristics is time-consuming and error-prone. The goal of this task is to engineer an autonomous Artificial Intelligence classification system that accepts structured physical feature inputs (petal and sepal dimensions) and accurately maps them to their correct botanical category using an artificial neural network.
2. Dataset Used
The assignment utilizes the classic Iris Dataset. It holds 150 distinct sample cases consisting of 4 continuous numeric inputs (Sepal Length, Sepal Width, Petal Length, Petal Width). The single prediction target maps to 3 target flower classes: Iris-setosa, Iris-versicolor, and Iris-virginica.
3. Framework Used
PyTorch (torch): Chosen for its explicit, hands-on, and highly Pythonic configuration workflow.
4. Model Architecture
The network is built as an explicit Multi-Layer Perceptron (MLP) containing the following sequential data pipeline design:
Input Layer: Accepts 4 floating-point features corresponding to the physical plant dimensions.
Hidden Layer 1: Maps 4 inputs to 16 internal processing neurons using a linear transformation (nn.Linear).
Non-linear Activation: Passes hidden layer calculations through a Rectified Linear Unit (nn.ReLU) filter to break mathematical linearity and unlock complex boundary processing.
Output Layer: Maps the 16 hidden paths down into 3 scoring values corresponding to the target classification categories.
Optimization Framework: Utilizes Cross-Entropy Loss (nn.CrossEntropyLoss) paired with an Adam Optimization learning rate controller tuned to $0.01$.
5. Accuracy Achieved
The trained model achieved a highly stable categorization metric, delivering a test dataset validation accuracy between 96% and 100% (depending on the random state split variance). Precision, Recall, and F1-scores across all three species classes remained balanced, indicating a robust fit without bias.
6. Challenges Faced
Data Formatting Requirements: Raw structured dataframes cannot interface directly with PyTorch's mathematical backpropagation mechanisms. The features had to be manually encoded, scaled to prevent feature dominance, and explicitly converted into typed torch.float32 and torch.long Tensors.
Loss Function Tracking: Ensuring the target classes match the criteria expected by nn.CrossEntropyLoss. PyTorch requires targets as unencoded category indices rather than one-hot arrays, meaning data shape management had to be tracked carefully.
7. Key Learnings
The 5-Step Deep Learning Cycle: Understood the structural necessity of manual training cycles: clearing gradients via zero_grad(), executing forward passes, computing objective error distances, backpropagating gradients via backward(), and stepping adjustments forward via optimizer.step().
The Function of Non-Linearity: Mastered the concept that without an activation function like ReLU, stacking multiple linear layers collapses mathematically into simple linear regression. Non-linear layers are essential for mapping real-world complexities.