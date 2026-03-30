import pennylane as qml
from torch import nn
import torch

class GaussianDressedQuantumNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_modes = self.config.get('n_qubits', 1)  # Using 'n_qubits' variable name for number of modes
        self.q_depth = self.config.get('q_depth', 1)
        
        self.quantum_device = qml.device('default.gaussian', wires=self.n_modes) 
        
        # Calculate number of parameters:
        # Per mode per layer: 1 (Rotation) + 2 (Squeezing) + 2 (Displacement) = 5
        # Per pair of modes (if n_modes > 1): 2 (Beamsplitter) * n_modes = 2 * n_modes
        self.params_per_layer = self.n_modes * 5
        if self.n_modes > 1:
            self.params_per_layer += self.n_modes * 2
            
        num_params = self.q_depth * self.params_per_layer
        self.q_params = nn.Parameter(self.config.get('q_delta', 0.1) * torch.randn(num_params))

        # Traditional layers
        self.pre_net = nn.Linear(self.config['layers'][-1], self.n_modes)
        self.post_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, self.config["latent_dim_mlp"])
        )
        
        self.quantum_net = qml.QNode(self.quantum_nn, self.quantum_device, interface="torch", diff_method="best")
        self.device = torch.device("cuda:0" if config['use_cuda'] else "cpu")

    def export_circuit_text(self):
        """Return a static text diagram of the current quantum circuit."""
        sample_inputs = torch.zeros(self.n_modes, dtype=self.q_params.dtype, device=self.q_params.device)
        sample_weights = self.q_params.detach()
        return qml.draw(self.quantum_net)(sample_inputs, sample_weights)

    def forward(self, input_features):
        q_in = self.pre_net(input_features)

        # Skip PennyLane execution during graph tracing to avoid trace-only failures
        # with tensor iteration and tensor->NumPy conversions inside the QNode path.
        if torch.jit.is_tracing() or type(q_in).__name__ == "RecorderTensor":
            dummy_q_out = q_in[:, :1] * 0.0 
            return self.post_net(dummy_q_out)

        # Iterate over the batch.
        # Since we use default.gaussian without batch support out of the box, we loop over the batch.
        q_out = []
        for elem in q_in:
            q_out_elem = self.quantum_net(elem, self.q_params).float()
            # If there's only 1 mode, qml returns a scalar 0-d tensor, we need to make it 1-d
            if q_out_elem.dim() == 0:
                q_out_elem = q_out_elem.unsqueeze(0)
            q_out.append(q_out_elem.unsqueeze(0))
            
        q_out = torch.cat(q_out).to(self.device)
        return self.post_net(q_out)

    def quantum_nn(self, inputs, weights):
        # 1. Data encoding: displace each mode by the input features
        for i in range(self.n_modes):
            qml.Displacement(inputs[i], 0.0, wires=i)

        # 2. Variational layers
        idx = 0
        for _ in range(self.q_depth):
            # Single-mode Gaussian gates
            for i in range(self.n_modes):
                qml.Rotation(weights[idx], wires=i); idx += 1
                qml.Squeezing(weights[idx], weights[idx+1], wires=i); idx += 2
                qml.Displacement(weights[idx], weights[idx+1], wires=i); idx += 2
            
            # Two-mode entangling gates (Beamsplitter) if more than 1 mode
            if self.n_modes > 1:
                for i in range(self.n_modes):
                    wire_1 = i
                    wire_2 = (i + 1) % self.n_modes
                    qml.Beamsplitter(weights[idx], weights[idx+1], wires=[wire_1, wire_2])
                    idx += 2
        
        # 3. Measurement: return the expected value of the position quadrature (X)
        # default.gaussian only supports single measurements
        return qml.expval(qml.QuadX(0))
