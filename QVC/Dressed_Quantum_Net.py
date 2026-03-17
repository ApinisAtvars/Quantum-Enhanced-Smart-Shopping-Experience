# source: https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/#variational-quantum-circuit
import pennylane as qml
from torch import nn
import torch

config = {
        "n_qubits": 8,              # Number of neurons in final MLP layer
        "learning_rate": 0.0003,    # Same as NeuMF
        "batch_size": 256,          # Same as NeuMF
        "num_epochs": 10,           # For testing
        "q_depth": 6,               # Number of variational layers
        "gamma_lr_scheduler": 0.1,  # Learning rate reduction applier every 10 epochs
        "q_delta": 0.01,            # Initial spread of random quantum weights
    }

class DressedQuantumNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantum_device = qml.device('default.qubit', wires=config['n_qubits'])
        self.q_params = nn.Parameter(config['q_delta'] * torch.randn(config["q_depth"] * config["n_qubits"]))
        self.post_net = nn.Linear(config["n_qubits"], 1)
        self.quantum_net = qml.QNode(self.quantum_net, self.quantum_device)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input_features):
        q_out = torch.Tensor(0, self.config["n_qubits"])
        q_out = q_out.to(self.device)
        
        for elem in input_features:
            q_out_elem = torch.hstack(self.quantum_net(elem, self.q_params)).float().unsqueeze(0)
            q_out = torch.cat(q_out, q_out_elem)
        
        return self.post_net(q_out)

    
    def H_layer(self, n_qubits):
        """
        Layer of single-qubit Hadamard gates.
        """
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)
    
    def RY_layer(self, w):
        """
        Layer of parametrized qubit rotations around the y axis.
        """
        for idx, element in enumerate(w):
            qml.RY(element, wires=idx)
    
    def entangling_layer(self, nqubits):
        """
        Layer of CNOTs followed by another shifted layer of CNOT.
        """
        # In other words it should apply something like :
        # CNOT  CNOT  CNOT  CNOT...  CNOT
        #   CNOT  CNOT  CNOT...  CNOT
        for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
            qml.CNOT(wires=[i, i + 1])

    def quantum_net(self, q_input_features, q_weights_flat):
        """
        The variational quantum circuit.
        """

        # Reshape weights
        q_weights = q_weights_flat.reshape(self.config["q_depth"], self.config["n_qubits"])

        # Start from state |+> , unbiased w.r.t. |0> and |1>
        self.H_layer(self.config["n_qubits"])

        # Embed features in the quantum node
        self.RY_layer(q_input_features)

        # Sequence of trainable variational layers
        for k in range(self.config["q_depth"]):
            self.entangling_layer(self.config["n_qubits"])
            self.RY_layer(q_weights[k])

        # Expectation values in the Z basis
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(self.config["n_qubits"])]
        return tuple(exp_vals)
    
if __name__=="__main__":
    
    DressedQuantumNetwork(config)