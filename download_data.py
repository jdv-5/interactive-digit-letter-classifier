import os
import urllib.request

os.makedirs("data", exist_ok=True)

urls = {
    "data/mnist_train.csv": "https://huggingface.co/datasets/marhora/mnist-emnist-letter-digit/blob/main/data/mnist_train.csv",
    "data/mnist_test.csv": "https://huggingface.co/datasets/marhora/mnist-emnist-letter-digit/blob/main/data/mnist_test.csv",
    "data/emnist_train.csv": "https://huggingface.co/datasets/marhora/mnist-emnist-letter-digit/blob/main/data/emnist_letters_train.csv",
    'data/emnist_test.csv': "https://huggingface.co/datasets/marhora/mnist-emnist-letter-digit/blob/main/data/emnist_letters_test.csv"
}

for local_path, url in urls.items():
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
        print(f"{local_path} saved")
    else:
        print(f"The file {local_path} exists already.")

print("\nAll the datasets are in data ")