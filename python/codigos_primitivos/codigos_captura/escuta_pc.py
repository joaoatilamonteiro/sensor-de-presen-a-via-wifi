import argparse
import datetime
import uuid
import os

import serial
import serial.tools.list_ports

BAUD_PADRAO = 115200
tamanho_esperado = 384

cabecalho = "sessao_id,rotulo,nota,timestamp_pc,timestamp_ms_esp32,rssi,tamanho,csi_raw\n"

def parse_args():
    p = argparse.ArgumentParser(description="Captura CSI do ESP 32 via serial, um arquivo por sessao")
    p.add_argument("--rotulo", required=True, help = "rotulo da sessao")
    p.add_argument("--nota", default="", help ="nota livre para dizer sobre a sessao")
    p.add_argument("--serial", default=None, help="Porta serial")
    p.add_argument("--baud", type=int, default=BAUD_PADRAO, help=f"baud rate padrao {BAUD_PADRAO}")
    return p.parse_args()

def achar_porta():
    portas = list(serial.tools.list_ports.comports())
    if len(portas) == 1:
        return portas[0].device
    print("Não consegui escolher a porta sozinho. Portas encontradas:")
    for p in portas:
        print(f"  {p.device}  ({p.description})")
    print("Rode de novo com --serial <porta>.")
    raise SystemExit(1)
 
 
def linha_valida(linha):
    """Espera: timestamp_ms,rssi,tamanho,csi_raw (384 valores). Descarta lixo de boot,
    cabeçalho, prints de outros módulos e linhas truncadas."""
    partes = linha.split(",")
    if len(partes) < 4:
        return False
    try:
        int(partes[0])
        int(partes[1])
        tamanho = int(partes[2])
        valores = [int(v) for v in partes[3:]]
    except ValueError:
        return False
    return tamanho == tamanho_esperado and len(valores) == tamanho
tamanho_esperado
 
def main():
    args = parse_args()
 
    porta = args.serial or achar_porta()
    sessao_id = uuid.uuid4().hex[:8]
    agora_arquivo = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arquivo_saida = f"csi_{args.rotulo}_{agora_arquivo}_{sessao_id}.csv"
 
    ser = serial.Serial(porta, args.baud, timeout=1)
    ser.reset_input_buffer()
 
    print(f"Sessão: {sessao_id} | rótulo: {args.rotulo} | nota: {args.nota!r}")
    print(f"Lendo {porta} a {args.baud} baud... Gravando em {arquivo_saida}")
    print("Ctrl+C para encerrar a sessão.")
 
    contador = 0
    descartadas = 0
 
    with open(arquivo_saida, "w", buffering=1) as f:
        f.write(cabecalho)
        try:
            while True:
                bruto = ser.readline()
                if not bruto:
                    continue  # timeout sem dados
 
                linha = bruto.decode("utf-8", errors="replace").strip()
                if not linha_valida(linha):
                    descartadas += 1
                    continue
 
                agora = datetime.datetime.now().isoformat(timespec="milliseconds")
                f.write(f"{sessao_id},{args.rotulo},{args.nota},{agora},{linha}\n")
 
                contador += 1
                if contador % 50 == 0:
                    print(f"{contador} amostras gravadas ({descartadas} descartadas)...")
                #if contador == 1000:
                    #ser.close
                    #os.system("shutdown /s /t 5")
        except KeyboardInterrupt:
            print(f"\nSessão {sessao_id} encerrada.")
            print(f"Total gravado: {contador} | Descartadas: {descartadas}")
            print(f"Arquivo: {arquivo_saida}")
        finally:
            ser.close()
 
 
if __name__ == "__main__":
    main()