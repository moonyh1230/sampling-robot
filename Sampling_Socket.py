import socket
import numpy as np


M_SIZE = 1024


class SocketClass:
    def __init__(self, name):
        self.udp_name = name
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_msg = None

    def __str__(self):
        return self.udp_name

    def send_messages(self, msg, ip, port):
        serv_addr = (ip, port)
        try:
            self.socket.sendto(msg, serv_addr)
            # serv_msg, addr = self.socket.recvfrom(M_SIZE)

            # self.recv_msg = serv_msg.decode(encoding="utf-8")

        except Exception as e:
            print(e)

    # def receive_messages(self, ip, port):

    def close_socket(self):
        self.socket.close()
