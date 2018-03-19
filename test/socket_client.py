import socket
import sys

HOST, PORT = "localhost", 9999
# data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a TCP socket)
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

for d in range(12, 100):
    data = str(d)
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

    print("Sent:     {}".format(send_data))
    print("Received: {}".format(received))
