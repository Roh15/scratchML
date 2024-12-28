import numpy as np
from tqdm import tqdm


class Sequential:
    def __init__(self, loss_fn):
        self.layers = []
        self.loss_fn = loss_fn
        self.output = None

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        return x

    def backward(self, y):
        # Perform a forward pass
        y_hat = self.output

        # Compute the loss and its gradient
        loss = self.loss_fn.forward(y, y_hat)
        grad = self.loss_fn.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return np.mean(loss)  # Return the mean loss for monitoring

    def train(self, epochs, train_loader, val_loader=None, val_freq=None):
        loss = {'train': []}
        if val_loader:
            loss['validation'] = []

        for epoch in range(epochs):
            train_loss = 0
            n_batches = 0

            # Single progress bar for current epoch
            train_batch_bar = tqdm(train_loader,
                                   desc=f'Epoch {epoch + 1}/{epochs} Train',
                                   leave=True)

            for train_data, train_targets in train_batch_bar:
                y_hat = self.forward(train_data)
                batch_loss = self.backward(train_targets)
                train_loss += batch_loss
                n_batches += 1

                # Update progress bar with running average loss
                current_avg_loss = train_loss / n_batches
                train_batch_bar.set_postfix({
                    'avg_loss': f'{current_avg_loss:.4f}'
                })

            # Calculate and store average epoch loss
            avg_train_loss = train_loss / len(train_loader)
            loss['train'].append(avg_train_loss)

            if val_loader:
                if (epoch + 1) % val_freq == 0 or epoch == epochs - 1:
                    val_loss = 0
                    n_batches = 0

                    val_batch_bar = tqdm(val_loader,
                                         desc=f'Val',
                                         leave=True,
                                         colour='green')

                    for val_data, val_targets in val_batch_bar:
                        y_hat = self.forward(val_data)
                        batch_loss = np.mean(self.loss_fn.forward(val_targets, y_hat))
                        val_loss += batch_loss
                        n_batches += 1

                        # Update progress bar with running average loss
                        current_avg_loss = val_loss / n_batches
                        val_batch_bar.set_postfix({
                            'avg_loss': f'{current_avg_loss:.4f}'
                        })

                    # Calculate and store average epoch loss
                    avg_val_loss = val_loss / len(val_loader)
                    loss['validation'].append(avg_val_loss)

        return loss

    def test(self, test_loader):
        test_loss = 0
        n_batches = 0

        test_batch_bar = tqdm(test_loader,
                              desc=f'Test',
                              leave=True,
                              colour='red')

        for test_data, test_targets in test_batch_bar:
            y_hat = self.forward(test_data)
            batch_loss = np.mean(self.loss_fn.forward(test_targets, y_hat))
            test_loss += batch_loss
            n_batches += 1

            # Update progress bar with running average loss
            current_avg_loss = test_loss / n_batches
            test_batch_bar.set_postfix({
                'avg_loss': f'{current_avg_loss:.4f}'
            })

        # Calculate and store average epoch loss
        avg_test_loss = test_loss / len(test_loader)

        return avg_test_loss

    def predict(self, X):
        y_hat = self.forward(X)
        return y_hat
