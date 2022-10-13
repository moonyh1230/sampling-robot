import socket
import struct
from Sampling_Socket import Receiver, Sender

import numpy as np


def run():
    sender_ip = "169.254.84.185"
    receiver_ip = "161.122.55.149"  # L3 main computer ip

    sender_port = 61456
    receiver_port = 61440

    udp_receiver = Receiver("udp_receiver", sender_ip, sender_port)
    udp_sender = Sender("udp_sender", receiver_ip, receiver_port)

    udp_receiver.start()

    while True:
        try:
            send_pack = []
            for i in range(3):
                num = input("insert number\n")
                send_pack.append(num)

            send_pack = list(map(int, send_pack))

            send = struct.pack("fff", send_pack[0], send_pack[1], send_pack[2])

            udp_sender.send_messages(send)
            print(send)

            recv = udp_receiver.get_recv_data()
            if recv is not None:
                print(recv)

            # decoded = struct.unpack("<fff", recv)
            #
            # for i in decoded:
            #     print(i)

        except KeyboardInterrupt:
            udp_receiver.close_receiver_socket()
            udp_sender.close_sender_socket()

            break


if __name__ == '__main__':
    run()
