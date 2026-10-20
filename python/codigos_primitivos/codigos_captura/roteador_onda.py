import socket
import time

# Substitua pelo IP que aparece no terminal do VS Code quando o ESP32 liga
IP_ESP32 = "192.168.1.165" 
PORTA = 5555

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"mandando 50 ondas/segundo para {IP_ESP32}...")
print("Deixe esta janela aberta a rodar enquanto grava os seus CSVs!")

try:
    while True:
        sock.sendto(b"radar", (IP_ESP32, PORTA))
        time.sleep(0.02)  # 20ms de delay (50 pacotes por segundo)
except KeyboardInterrupt:
    print("\nDisparos interrompidos.")