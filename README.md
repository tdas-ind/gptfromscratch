# gptfromscratch

Step 1 : To create env conda create -p <env name> python==3.10 -y

Step 2 : Activate env conda activate/<env name>

Step 3 : Install the requirements.txt "pip install -r requirements.txt"

Step 4 : Download input.txt which is the input where the model is trained



######################

A basic character-level decoder-based neural network model trained on Shakespeare's text. The model uses multi-head attention, self-attention, feed-forward network, layer normalization, and residual connection to generate text at the character level

Architecture

Encoder: self-attention + feed-forward network /n
Decoder: multi-head attention + self-attention + feed-forward network
Multi-head Attention: computes attention weights across different heads
Self-Attention: computes attention weights within the same sequence
Feed Forward Network: fully connected feed-forward network
Layer Normalization: normalizes output to have mean 0 and std 1
Residual Connection: adds output to input to learn residual functions

Acknowledgments

This model was created with the help of Andrej Karpathy's lecture on attention mechanisms. Huge shoutout to him for his contributions to the field of deep learning!
