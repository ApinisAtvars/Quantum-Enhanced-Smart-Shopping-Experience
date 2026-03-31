# source: https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/#variational-quantum-circuit
import pennylane as qml
from torch import nn
import torch

class DressedQuantumNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantum_device = qml.device('lightning.qubit', wires=self.config['n_qubits'])
        num_params = self.config["q_depth"] * self.config["n_qubits"] * 3
        self.q_params = nn.Parameter(self.config['q_delta'] * torch.randn(num_params))

        # Traditional layers
        self.pre_net = nn.Linear(self.config['layers'][-1], self.config['n_qubits'])
        self.post_net = nn.Sequential(
            nn.Linear(self.config["n_qubits"], 16),
            nn.ReLU(),
            nn.Linear(16, self.config["latent_dim_mlp"])
        )
        
        self.quantum_net = qml.QNode(self.strongly_entangling_layers, self.quantum_device, interface="torch", diff_method="best")
        self.device = torch.device(f"cuda:{config['device_id']}" if config['use_cuda'] else "cpu")

    def export_circuit_text(self):
        """Return a static text diagram of the current quantum circuit."""
        sample_inputs = torch.zeros(self.config["n_qubits"], dtype=self.q_params.dtype, device=self.q_params.device)
        sample_weights = self.q_params.detach()
        return qml.draw(self.quantum_net)(sample_inputs, sample_weights)


    def forward(self, input_features):
        q_in = self.pre_net(input_features)

        # Skip PennyLane execution during tracing because QNode internals rely on
        # Python-side shape checks and tensor conversions that are not trace-safe.
        if torch.jit.is_tracing() or type(q_in).__name__ == "RecorderTensor":
            return self.post_net(q_in * 0.0)

        q_out = []
        for elem in q_in:
            q_out_elem = torch.hstack(self.quantum_net(elem, self.q_params)).to(dtype=q_in.dtype)
            q_out.append(q_out_elem.unsqueeze(0))

        q_out = torch.cat(q_out, dim=0).to(q_in.device)
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

    # 1st option from source (1st line)
    # Not very good
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
    
    # 2nd option StronglyEntanglingLayers & AngleEmbedding (commented out in the code below)
    def strongly_entangling_layers(self, q_input_features, q_weights_flat):
        """
        The variational quantum circuit using StronglyEntanglingLayers.
        """
        # 1. Reshape weights to (q_depth, n_qubits, 3)
        q_weights = q_weights_flat.reshape(self.config["q_depth"], self.config["n_qubits"], 3)
        
        # 2. Embed features using AngleEmbedding
        qml.AngleEmbedding(features=q_input_features, wires=range(self.config["n_qubits"]), rotation='Y')

        # 3. Apply the Trainable Ansatz
        qml.StronglyEntanglingLayers(weights=q_weights, wires=range(self.config["n_qubits"]))

        # 4. Measure
        exp_vals =[qml.expval(qml.PauliZ(position)) for position in range(self.config["n_qubits"])]
        return tuple(exp_vals)
    
if __name__=="__main__":
    pass