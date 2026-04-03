# source: https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning/#variational-quantum-circuit
import pennylane as qml
from torch import nn
import torch

class DressedQuantumNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.circuit_type = config.get('circuit_type', 'strongly_entangling')
        self.use_residual = config.get('use_residual', False)
        
        self.quantum_device = qml.device('lightning.qubit', wires=self.config['n_qubits'])
        
        # Determine number of parameters and quantum function based on circuit type
        if self.circuit_type == 'strongly_entangling':
            num_params = self.config["q_depth"] * self.config["n_qubits"] * 3
            q_func = self.strongly_entangling_layers
        elif self.circuit_type == 'data_reuploading':
            num_params = self.config["q_depth"] * self.config["n_qubits"] * 2
            q_func = self.data_reuploading_circuit
        elif self.circuit_type == 'iqp':
            num_params = self.config["q_depth"] * self.config["n_qubits"]
            q_func = self.iqp_circuit
        elif self.circuit_type == 'mps':
            num_params = self.config["q_depth"] * self.config["n_qubits"]
            q_func = self.mps_circuit
        else:
            raise ValueError(f"Unknown circuit type: {self.circuit_type}")
            
        self.q_params = nn.Parameter(self.config['q_delta'] * torch.randn(num_params))

        # Traditional layers
        self.pre_net = nn.Linear(self.config['layers'][-1], self.config['n_qubits'])
        self.post_net = nn.Sequential(
            nn.Linear(self.config["n_qubits"], 16),
            nn.ReLU(),
            nn.Linear(16, self.config["latent_dim_mlp"])
        )
        
        self.quantum_net = qml.QNode(q_func, self.quantum_device, interface="torch", diff_method="best")
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
        
        if self.use_residual:
            q_out = q_out + q_in
            
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

    def data_reuploading_circuit(self, q_input_features, q_weights_flat):
        """
        Interleaves data embedding with trainable rotations.
        Uses 2 parameters per qubit per layer (RY and RZ).
        """
        q_weights = q_weights_flat.reshape(self.config["q_depth"], self.config["n_qubits"], 2)
        for layer in range(self.config["q_depth"]):
            qml.AngleEmbedding(features=q_input_features, wires=range(self.config["n_qubits"]), rotation='Y')
            for i in range(self.config["n_qubits"]):
                qml.RY(q_weights[layer, i, 0], wires=i)
                qml.RZ(q_weights[layer, i, 1], wires=i)
            for i in range(self.config["n_qubits"] - 1):
                qml.CZ(wires=[i, i + 1])
            qml.CZ(wires=[self.config["n_qubits"] - 1, 0])
        return tuple([qml.expval(qml.PauliZ(position)) for position in range(self.config["n_qubits"])])

    def iqp_circuit(self, q_input_features, q_weights_flat):
        """
        Uses IQPEmbedding to capture complex feature cross-correlations.
        """
        q_weights = q_weights_flat.reshape(self.config["q_depth"], self.config["n_qubits"])
        qml.IQPEmbedding(features=q_input_features, wires=range(self.config["n_qubits"]))
        qml.BasicEntanglingLayers(weights=q_weights, wires=range(self.config["n_qubits"]))
        return tuple([qml.expval(qml.PauliZ(position)) for position in range(self.config["n_qubits"])])

    def mps_circuit(self, q_input_features, q_weights_flat):
        """
        Matrix Product State (MPS) inspired layer. 
        Highly resilient to barren plateaus.
        """
        q_weights = q_weights_flat.reshape(self.config["q_depth"], self.config["n_qubits"])
        qml.AngleEmbedding(features=q_input_features, wires=range(self.config["n_qubits"]), rotation='Y')
        for k in range(self.config["q_depth"]):
            # Sweep left to right (MPS style)
            for i in range(self.config["n_qubits"] - 1):
                qml.RY(q_weights[k, i], wires=i)
                qml.RY(q_weights[k, i+1], wires=i+1)
                qml.CNOT(wires=[i, i+1])
        return tuple([qml.expval(qml.PauliZ(position)) for position in range(self.config["n_qubits"])])

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
