import socket
import struct
from Sampling_Socket import Receiver, Sender

import numpy as np


def run():
    sender_ip = "192.168.8.3"
    receiver_ip = "192.168.8.1"  # L3 main computer ip

    sender_port = 61496
    receiver_port = 61480

    udp_receiver = Receiver("udp_receiver", sender_ip, sender_port)
    udp_sender = Sender("udp_sender", receiver_ip, receiver_port)

    udp_receiver.start()

    while True:
        try:
            send_pack = []
            for i in range(8):
                num = input("insert number\n")
                send_pack.append(num)

            send_pack = list(map(int, send_pack))

            send = struct.pack("ffffffff", send_pack[0], send_pack[1], send_pack[2], send_pack[3], send_pack[4], send_pack[5], send_pack[6], send_pack[7])

            udp_sender.send_messages(send)
            print(send)

            recv = udp_receiver.get_recv_data()

            if recv is not None:
                clustered_data = recv.decode('utf-8')
                listed = list(map(int, clustered_data.split('[')[1].split(']')[0].split(',')))

                print(listed)

        except KeyboardInterrupt:
            udp_receiver.close_receiver_socket()
            udp_sender.close_sender_socket()

            break


if __name__ == '__main__':
    run()
