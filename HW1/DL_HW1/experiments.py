import numpy as np
import itertools
from network_template import Network, load_data_cifar, cross_entropy
import sys


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


# sys.stdout = Tee("results.txt")
train_file = "./data/train_data.pckl"
test_file = "./data/test_data.pckl"
train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
print(train_data.shape)
exit()

val_pct = 0.1
val_size = int(train_data.shape[1] * val_pct)
val_data = train_data[..., :val_size]
val_class = train_class[..., :val_size]
# train_data = train_data[..., val_size:]
# train_class = train_class[..., val_size:]

INPUT_SIZE = train_data.shape[0]
OUTPUT_SIZE = 10


def evaluate(net, data, labels):
    tp = 0
    loss_avg = 0.0
    n = data.shape[1]
    for i in range(n):
        x = np.expand_dims(data[:, i], -1)
        y = np.expand_dims(labels[:, i], -1)
        out, _, _ = net.forward_pass(x)
        tp += int(np.argmax(out) == np.argmax(labels[:, i]))
        loss_avg += cross_entropy(y, out)
    return tp / n, loss_avg / n


def grid_search():
    print("\n" + "=" * 60)
    print("TASK 1 — Grid Search")
    print("=" * 60)

    hidden_configs = [
        [256],
        [256, 128],
        [512, 256],
        [256, 128, 64],
    ]
    learning_rates = [0.001, 0.01]
    batch_sizes = [64, 128]
    epochs = 10

    results = []

    for hidden, lr, bs in itertools.product(
        hidden_configs, learning_rates, batch_sizes
    ):
        sizes = [INPUT_SIZE] + hidden + [OUTPUT_SIZE]
        net = Network(sizes, optimizer="adam", lambda_=0.0)
        net.train(
            train_data,
            train_class,
            val_data,
            val_class,
            epochs=epochs,
            mini_batch_size=bs,
            eta=lr,
        )

        acc, loss = evaluate(net, val_data, val_class)
        config = {
            "hidden": hidden,
            "lr": lr,
            "batch_size": bs,
            "val_acc": round(acc, 4),
            "val_loss": round(loss, 4),
        }
        results.append(config)
        print(f"  hidden={hidden}  lr={lr}  bs={bs} -  val_acc={acc:.4f}")

    results.sort(key=lambda r: r["val_acc"], reverse=True)
    print("\nTop 3 configs:")
    for r in results[:3]:
        print(" ", r)

    best = results[0]
    print(f"\nBest config: {best}")
    return best


def adam_vs_sgd(best_config):
    print("\n" + "=" * 60)
    print("TASK 2 — Adam vs SGD")
    print("=" * 60)

    hidden = best_config["hidden"]
    bs = best_config["batch_size"]
    epochs = 20
    sizes = [INPUT_SIZE] + hidden + [OUTPUT_SIZE]

    experiments = [
        ("adam", 0.001),
        ("sgd", 0.01),
    ]

    for optimizer, lr in experiments:
        print(f"\n  Optimizer={optimizer}  lr={lr}")
        net = Network(sizes, optimizer=optimizer, lambda_=0.0)
        net.train(
            train_data,
            train_class,
            val_data,
            val_class,
            epochs=epochs,
            mini_batch_size=bs,
            eta=lr,
        )
        acc, loss = evaluate(net, test_data, test_class)
        print(f"  Test acc={acc:.4f}  Test loss={loss:.4f}")


def regularisation(best_config):
    print("\n" + "=" * 60)
    print("TASK 3 — L2 Regularisation")
    print("=" * 60)

    hidden = best_config["hidden"]
    bs = best_config["batch_size"]
    lr = best_config["lr"]
    epochs = 20
    sizes = [INPUT_SIZE] + hidden + [OUTPUT_SIZE]

    experiments = [
        ("adam", 0.0, "Adam  no L2"),
        ("adam", 0.001, "Adam  L2=0.001"),
        ("sgd", 0.0, "SGD   no L2"),
        ("sgd", 0.001, "SGD   L2=0.001"),
    ]

    lr_map = {"adam": 0.001, "sgd": 0.01}

    for optimizer, lambda_, label in experiments:
        print(f"\n  {label}")
        net = Network(sizes, optimizer=optimizer, lambda_=lambda_)
        net.train(
            train_data,
            train_class,
            val_data,
            val_class,
            epochs=epochs,
            mini_batch_size=bs,
            eta=lr_map[optimizer],
        )
        acc, loss = evaluate(net, test_data, test_class)
        print(f"  Test acc={acc:.4f}  Test loss={loss:.4f}")


def lr_schedule(best_config):
    print("\n" + "=" * 60)
    print("TASK 4 — Learning Rate Schedule")
    print("=" * 60)

    hidden = best_config["hidden"]
    bs = best_config["batch_size"]
    epochs = 20
    sizes = [INPUT_SIZE] + hidden + [OUTPUT_SIZE]

    experiments = [
        (False, 0.9, "Adam  no schedule"),
        (True, 0.05, "Adam  exp decay=0.05"),
        (True, 0.1, "Adam  exp decay=0.1"),
    ]

    for decay, decay_rate, label in experiments:
        print(f"\n  {label}")
        net = Network(
            sizes, optimizer="adam", lambda_=0.001, decay=decay, decay_rate=decay_rate
        )
        net.train(
            train_data,
            train_class,
            val_data,
            val_class,
            epochs=epochs,
            mini_batch_size=bs,
            eta=0.001,
        )
        acc, loss = evaluate(net, test_data, test_class)
        print(f"  Test acc={acc:.4f}  Test loss={loss:.4f}")


if __name__ == "__main__":

    # best_config = grid_search()
    best_config = {"hidden": [256], "lr": 0.001, "batch_size": 128}
    adam_vs_sgd(best_config)
    regularisation(best_config)
    lr_schedule(best_config)
