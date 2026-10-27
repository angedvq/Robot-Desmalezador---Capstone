import pygame, time
import serial, struct

# Init
ser = serial.Serial("/dev/ttyAMA1", 115200)


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
    print(f"Dispositivo: {js.get_name()} | ejes: {js.get_numaxes()} | botones: {js.get_numbuttons()}")

    while True:
        pygame.event.pump()

        eje_x = round(js.get_axis(0) * (-1/0.6), 1) * 180

        print(f"Eje x: {eje_x}")
        enviar(1, eje_x)

        eje_y = round(js.get_axis(1) * (-1 / 0.66), 1) * 0.369

        print(f"Eje y: {eje_y}")
        enviar(0, eje_y)

        time.sleep(0.01)


def debug():
    while True:
        valor = float(input("Ingresar valor:"))

        eje_x = valor * 180

        print(f"Eje x: {eje_x}")
        enviar(1, eje_x)

        eje_y = valor * 0.369

        print(f"Eje y: {eje_y}")
        enviar(0, eje_y)


joystick()
