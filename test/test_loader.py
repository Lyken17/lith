import time
import socket
import sys

import torch
import torch.utils.data as data

from torchvision.datasets import cifar

from multiprocessing import Process, Manager, Value

x = Value('i', 0)

import threading

class FakeDataset(data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return int(1e6)

    def __getitem__(self, index):


        HOST, PORT = "localhost", 9999
        # data = " ".join(sys.argv[1:])

        # Create a socket (SOCK_STREAM means a TCP socket)
        # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        data = str(index)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Connect to server and send data
            sock.connect((HOST, PORT))
            send_data = (data + "\n").encode()
            sock.sendall(send_data)

            # Receive data from the server and shut down
            received = sock.recv(1024)
        finally:
            sock.close()


        return (torch.zeros(3, 3, 1, 1), 1)


if __name__ == "__main__":
    dst = FakeDataset()
    end = time.time()

    socket.setdefaulttimeout(2000)

    num_workers = 6
    dst = data.DataLoader(dst, num_workers=num_workers)

    for idx, item in enumerate(dst):
        time.sleep(1.5)

        elapased = time.time() - end
        if idx % num_workers == 0:
            print()
        print("Index: %d Time: %.6f" % (idx, elapased))

        end = time.time()
