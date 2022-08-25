import socket
import struct

import numpy


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind(('127.0.0.1', 7000))
while True:
    recv, addr = sock.recvfrom(200)

    decoded = struct.unpack("<fff", recv)

    for i in decoded:
        print(i)
