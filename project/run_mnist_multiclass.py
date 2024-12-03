# from mnist import MNIST

import minitorch


import os
import urllib.request
import gzip

def download_mnist():
    """Download MNIST dataset if not present"""
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and extract each file
    for filename in files:
        filepath = os.path.join(data_dir, filename[:-3])  # Remove .gz
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, 
                                     os.path.join(data_dir, filename))
            # Extract .gz file
            with gzip.open(os.path.join(data_dir, filename), 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    f_out.write(f_in.read())
            # Remove .gz file
            os.remove(os.path.join(data_dir, filename))

# Add this before the MNIST loading code
download_mnist()
data_dir = os.path.join(os.path.dirname(__file__), "data")
mndata = MNIST(data_dir)
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        # Get batch size and input dimensions
        batch, in_channels, h, w = input.shape
        out_channels, _, kh, kw = self.weights.value.shape
        
        # Calculate output dimensions
        out_h = h - kh + 1
        out_w = w - kw + 1
        
        # Initialize output tensor
        output = input.zeros((batch, out_channels, out_h, out_w))
        
        # Perform convolution manually
        for b in range(batch):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Manual convolution for each window
                        total = 0.0
                        for ic in range(in_channels):
                            for ki in range(kh):
                                for kj in range(kw):
                                    in_val = input[b, ic, i + ki, j + kj]
                                    weight_val = self.weights.value[oc, ic, ki, kj]
                                    total += in_val * weight_val
                        output[b, oc, i, j] = total + self.bias.value[oc, 0, 0]
        
        return output


class Network(minitorch.Module):
    def __init__(self):
        super().__init__()
        
        # For vis
        self.mid = None
        self.out = None
        
        # First convolution layer (1 input channel, 4 output channels, 3x3 kernel)
        self.conv1 = Conv2d(1, 4, 3, 3)
        
        # Second convolution layer (4 input channels, 8 output channels, 3x3 kernel)
        self.conv2 = Conv2d(4, 8, 3, 3)
        
        # Calculate dimensions after convolutions and pooling
        # Input: 28x28
        # After conv1 (3x3): 26x26
        # After conv2 (3x3): 24x24
        # After maxpool (4x4): 6x6
        # Flattened size: 8 * 6 * 6 = 288
        self.flat_size = 8 * 6 * 6
        
        # Linear layers
        self.linear1 = Linear(self.flat_size, 64)
        self.linear2 = Linear(64, C)    # C = number of classes (10)

    def forward(self, x):
        # First conv + ReLU
        self.mid = self.conv1.forward(x).relu()
        
        # Second conv + ReLU
        self.out = self.conv2.forward(self.mid).relu()
        
        # Max pooling with 4x4 kernel
        pooled = minitorch.nn.maxpool2d(self.out, (4, 4))
        
        # Flatten - explicitly specify both dimensions
        batch_size = pooled.shape[0]
        flattened = pooled.view(batch_size, self.flat_size)
        
        # First linear layer + ReLU + Dropout
        hidden = self.linear1.forward(flattened).relu()
        hidden = minitorch.nn.dropout(hidden, 0.25)
        
        # Final linear layer
        logits = self.linear2.forward(hidden)
        
        # Log softmax
        return minitorch.nn.logsoftmax(logits, dim=1)


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
