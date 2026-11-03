import pygame
import time
import serial
import struct

# Init
ser = serial.Serial("COM10", 115200)


def enviar(orden, valor):
    valor = min(valor, 327.67)
    valor = max(valor, -327.67)

    v = int(valor * 100)
    payload = struct.pack("<Bh", orden, v)
    checksum = payload[0] ^ payload[1] ^ payload[2] ^ 0xA5
    ser.write(payload + bytes([checksum]))


def joystick():
    pygame.init()
    pygame.joystick.init()

    js = pygame.joystick.Joystick(0)

    while True:
        pygame.event.pump()

        eje_x = round(js.get_axis(0), 1)
        if abs(eje_x) < 0.1:
            eje_x = 0
        enviar(1, eje_x * 180)

        eje_y = round(js.get_axis(1), 1)
        if abs(eje_y) < 0.1:
            eje_y = 0
        enviar(0, eje_y * -0.369)

        print(f"{eje_x} - {eje_y}")

        hx, hy = js.get_hat(0)
        if hx == 1:
            enviar(5, -100)
            print("Derecha")

        if hx == -1:
            enviar(5, 100)
            print("Izquierda")

        if hy == -1:
            enviar(6, 0)
            print("Abajo")

        a = js.get_button(0)
        if a == 1:
            enviar(7, 0)
            print("boton A")

        b = js.get_button(1)
        if b == 1:
            enviar(8,  0)
            print("boton B")

        time.sleep(0.1)


joystick()
