"""
Demo ao vivo de deteccao de MOVIMENTO por CSI (ESP32 -> serial -> PC).  v2

Como funciona (v3):
  1. CALIBRACAO: N segundos com a area do link parada. Aprende, para CADA
     subportadora, o quanto ela oscila "em repouso" naquele lugar e momento.
  2. AO VIVO: a cada 0,5 s mede 5 numeros do ultimo 1 s (em unidades do repouso do dia).
       - sem --modelo: compara a oscilacao com um limiar (normal + k * ruido);
       - com --modelo: uma ARVORE DE DECISAO treinada (treinar_arvore.py) decide.
     Alerta quando pelo menos M das ultimas N leituras acusam movimento.

Uso:
  python demo_ao_vivo.py --ip 192.168.1.165
  python demo_ao_vivo.py --ip 192.168.1.165 --serial COM13 --calibracao 60
  python demo_ao_vivo.py --ip 192.168.1.165 --modelo modelo_arvore.joblib   (decide com a arvore)
  python demo_ao_vivo.py --simular                       (ensaio sem hardware)
  python demo_ao_vivo.py --replay vazio.csv --alvo andando.csv --texto
                                                         (reexibe gravacoes reais)

Teclas na janela:  C = recalibrar | + / - = sensibilidade | Q = sair
Requisitos: pip install pyserial numpy matplotlib pandas scikit-learn joblib
Precisa de csi_features.py na mesma pasta.
"""

import argparse
import csv
import datetime
import socket
import threading
import time
from collections import deque

import numpy as np

import csi_features as cf

TAMANHO = 384        # bytes de CSI por pacote
N_SUB = cf.N_SUB
MIN_PKTS = cf.MIN_PKTS
JANELA_MS = cf.JANELA_MS
PASSO_MS = cf.PASSO_MS


# ----------------------------------------------------------------------------
# Pacotes
# ----------------------------------------------------------------------------
def parse_linha(linha):
    """'timestamp_ms,rssi,tamanho,v0,...,v383' -> (ts, amplitude[64]) ou None."""
    p = linha.split(",")
    if len(p) != 3 + TAMANHO:
        return None
    try:
        ts = int(p[0])
        if int(p[2]) != TAMANHO:
            return None
        v = np.array(p[3:], dtype=np.float32)
    except ValueError:
        return None
    return ts, np.hypot(v[0::2], v[1::2])[:N_SUB]


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
class Detector:
    def __init__(self, calib_s=60.0, sensib=1.0, k=3.0, n=10, m=4, modelo=None):
        self.calib_s, self.sensib, self.k, self.n, self.m = calib_s, sensib, k, n, m
        self.modelo = modelo
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.fase = "calibrando"
        self.calib = []
        self.base = None
        self.med = self.sigma = None
        self.janela = deque()
        self.ultimo_calc = None
        self.ultimo_ts = None
        self.flags = deque(maxlen=self.n)
        self.historico = deque(maxlen=240)
        self.valor = None
        self.alerta = False
        self.pps = 0
        self.prog = 0.0
        self.msg = ""
        self.total = 0

    @property
    def limiar(self):
        if self.modelo is not None:                       # arvore: limiar de probabilidade
            return float(np.clip(0.5 / self.sensib, 0.05, 0.95))
        return self.med + (self.k / self.sensib) * self.sigma

    def _finalizar_calibracao(self):
        ts = np.array([t for t, _ in self.calib], dtype=np.float64)
        arr = np.stack([a for _, a in self.calib])
        pps = len(ts) / max((ts[-1] - ts[0]) / 1000.0, 1e-9)
        base, F = cf.calibrar(ts, arr)
        if base is None:
            self.msg = f"Calibracao falhou: so {pps:.0f} pacotes/s. Tentando de novo..."
            self.calib = []
            return
        self.base = base
        z = F[:, 0]
        self.med = float(np.median(z))
        self.sigma = max(1.4826 * float(np.median(np.abs(z - self.med))), 0.05 * self.med)
        self.fase = "ao_vivo"
        aviso = "  [AVISO: taxa baixa, o ideal e ~50/s]" if pps < 35 else ""
        modo = "ARVORE" if self.modelo is not None else f"limiar={self.med + self.k * self.sigma:.3f}"
        self.msg = f"Calibrado: {pps:.0f} pacotes/s | modo {modo}{aviso}"

    def add(self, ts, amp):
        """Processa um pacote. Devolve True quando calculou um valor novo."""
        with self.lock:
            self.total += 1
            if self.ultimo_ts is not None and ts < self.ultimo_ts:   # ESP32 reiniciou
                self.janela.clear()
                self.ultimo_calc = None
                self.calib = []
            self.ultimo_ts = ts

            if self.fase == "calibrando":
                self.calib.append((ts, amp))
                dec = (ts - self.calib[0][0]) / 1000.0
                self.prog = min(dec / self.calib_s, 1.0)
                if dec >= self.calib_s:
                    self._finalizar_calibracao()
                return False

            self.janela.append((ts, amp[self.base.mask]))
            while self.janela and self.janela[0][0] <= ts - JANELA_MS:
                self.janela.popleft()
            self.pps = len(self.janela)

            if self.ultimo_calc is None or ts - self.ultimo_calc >= PASSO_MS:
                self.ultimo_calc = ts
                if len(self.janela) >= MIN_PKTS:
                    raw = np.stack([r for _, r in self.janela])
                    x = raw / np.clip(raw.mean(axis=1, keepdims=True), 1e-9, None)
                    f = cf.features_janela(self.base, x, raw)
                    if self.modelo is not None:
                        self.valor = float(self.modelo.predict_proba(f[None, :])[0, 1])
                    else:
                        self.valor = float(f[0])
                    self.flags.append(self.valor > self.limiar)
                    self.alerta = sum(self.flags) >= self.m
                    self.historico.append(self.valor)
                    return True
                self.valor = None
            return False

    def snapshot(self):
        with self.lock:
            return dict(fase=self.fase, prog=self.prog, valor=self.valor, alerta=self.alerta,
                        pps=self.pps, msg=self.msg, sensib=self.sensib, total=self.total,
                        arvore=self.modelo is not None,
                        limiar=(self.limiar if self.fase == "ao_vivo" else None),
                        hist=list(self.historico))


# ----------------------------------------------------------------------------
# Fontes de pacotes (cada uma roda numa thread e chama processar(ts, amp))
# ----------------------------------------------------------------------------
def achar_porta():
    import serial.tools.list_ports
    portas = list(serial.tools.list_ports.comports())
    chaves = ("CP210", "CH340", "USB-SERIAL", "USB SERIAL", "UART")
    cand = [p for p in portas if any(c in (p.description or "").upper() for c in chaves)]
    if len(cand) == 1:
        return cand[0].device
    print("Nao consegui escolher a porta sozinho. Portas encontradas:")
    for p in portas:
        print(f"  {p.device}  ({p.description})")
    raise SystemExit("Rode de novo com --serial <porta>.")


def fonte_serial(porta, baud, processar, parar, erros, contagem):
    """Leitura em blocos (bem mais rapida que readline) e buffer grande do Windows."""
    import serial
    try:
        ser = serial.Serial(porta, baud, timeout=0.1)
        try:
            ser.set_buffer_size(rx_size=1 << 20, tx_size=1 << 14)   # so existe no Windows
        except Exception:
            pass
        ser.reset_input_buffer()
        buf = bytearray()
        while not parar.is_set():
            dados = ser.read(max(ser.in_waiting, 1))
            if not dados:
                continue
            buf += dados
            while True:
                i = buf.find(b"\n")
                if i < 0:
                    break
                linha = bytes(buf[:i]).decode("utf-8", errors="replace").strip()
                del buf[:i + 1]
                pkt = parse_linha(linha)
                if pkt:
                    processar(*pkt)
                else:
                    contagem["invalidas"] += 1
            if len(buf) > 50000:      # lixo sem quebra de linha
                buf.clear()
        ser.close()
    except Exception as e:
        erros.append(f"Erro na serial: {e}")
        parar.set()


def disparar_udp(ip, porta, intervalo_s, parar):
    """Manda pacotes ao ESP32 em ritmo fixo (e o que gera o CSI)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    prox = time.perf_counter()
    while not parar.is_set():
        try:
            sock.sendto(b"radar", (ip, porta))
        except OSError:
            pass
        prox += intervalo_s
        atraso = prox - time.perf_counter()
        if atraso > 0:
            time.sleep(atraso)
        else:
            prox = time.perf_counter()
    sock.close()


def fonte_simulador(processar, parar, calib_s):
    rng = np.random.default_rng(0)
    base = rng.uniform(15, 40, 192)
    fase = rng.uniform(0, 2 * np.pi, 192)
    t0 = time.monotonic()
    prox = t0
    while not parar.is_set():
        t = time.monotonic() - t0
        mov = (calib_s + 8 <= t < calib_s + 14) or (calib_s + 22 <= t < calib_s + 28)
        ruido = 1.2 if mov else 0.3
        i = base * np.cos(fase) + rng.normal(0, ruido, 192)
        q = base * np.sin(fase) + rng.normal(0, ruido, 192)
        v = np.empty(384)
        v[0::2], v[1::2] = i, q
        v = np.clip(np.round(v), -128, 127).astype(int)
        pkt = parse_linha(f"{int(t * 1000)},-50,{TAMANHO}," + ",".join(map(str, v)))
        if pkt:
            processar(*pkt)
        prox += 0.02
        atraso = prox - time.monotonic()
        if atraso > 0:
            time.sleep(atraso)


def _pacotes_csv(arq, limite_s=None):
    import pandas as pd
    cols = [4] + list(range(7, 7 + TAMANHO))
    tipos = {c: np.float32 for c in cols}
    tipos[4] = np.float64
    t_ini = None
    for ch in pd.read_csv(arq, header=None, skiprows=1, usecols=cols, dtype=tipos,
                          chunksize=20000, on_bad_lines="skip", low_memory=False):
        ch = ch.dropna()
        if ch.empty:
            continue
        ts = ch.iloc[:, 0].to_numpy()
        v = ch.iloc[:, 1:].to_numpy(np.float32)
        amp = np.hypot(v[:, 0::2], v[:, 1::2])[:, :N_SUB]
        for j in range(len(ts)):
            if t_ini is None:
                t_ini = ts[j]
            if limite_s is not None and (ts[j] - t_ini) / 1000.0 > limite_s:
                return
            yield ts[j], amp[j]


def fonte_replay(arq_calib, arq_alvo, calib_s, velocidade, processar, parar, ao_fim):
    """Alimenta o detector com uma gravacao de calibracao e depois com outra."""
    ult = 0.0
    for ts, amp in _pacotes_csv(arq_calib, limite_s=calib_s + 3):
        if parar.is_set():
            return
        processar(ts, amp)
        ult = ts
    t0_alvo, wall0, ts_ini = None, time.perf_counter(), None
    for ts, amp in _pacotes_csv(arq_alvo):
        if parar.is_set():
            return
        if t0_alvo is None:
            t0_alvo, ts_ini = ts, ts
        ts_aj = ts - t0_alvo + ult + 1000
        if velocidade > 0:
            alvo = (ts - ts_ini) / 1000.0 / velocidade
            atraso = alvo - (time.perf_counter() - wall0)
            if atraso > 0.005:
                time.sleep(atraso)
        processar(ts_aj, amp)
    ao_fim()


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
class Janela:
    def __init__(self, det, parar):
        import matplotlib.pyplot as plt
        self.plt, self.det, self.parar = plt, det, parar
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 4.5), dpi=80)
        self.fig.subplots_adjust(top=0.80)
        self.titulo = self.fig.text(0.5, 0.93, "", ha="center", va="center",
                                    fontsize=22, fontweight="bold")
        self.linha, = self.ax.plot([], [], lw=2, color="#1f77b4")
        self.lim = self.ax.axhline(0, color="red", ls="--", label="limiar")
        self.ax.set_xlabel("segundos atras")
        self.ax.set_ylabel("oscilacao (1.0 = repouso)")
        self.ax.legend(loc="upper left")
        self.fig.canvas.mpl_connect("key_press_event", self.tecla)
        self.ultimo = 0.0

    def tecla(self, ev):
        if ev.key in ("q", "Q"):
            self.parar.set()
        elif ev.key in ("c", "C"):
            self.det.reset()
        elif ev.key in ("+", "="):
            self.det.sensib *= 1.15
        elif ev.key in ("-", "_"):
            self.det.sensib /= 1.15

    def aberta(self):
        return self.plt.fignum_exists(self.fig.number)

    def tick(self):
        """Redesenha so 2x por segundo (redesenhar sem parar trava a leitura)."""
        if time.time() - self.ultimo >= 0.5:
            self.ultimo = time.time()
            s = self.det.snapshot()
            if s["fase"] == "calibrando":
                self.titulo.set_text(f"CALIBRANDO {s['prog']:.0%}")
                self.titulo.set_color("#e69f00")
                self.ax.set_title("mantenha a area do link parada. " + s["msg"], fontsize=10)
            else:
                self.titulo.set_text("MOVIMENTO DETECTADO" if s["alerta"] else "SEM MOVIMENTO")
                self.titulo.set_color("#d62728" if s["alerta"] else "#2ca02c")
                val = f"{s['valor']:.2f}" if s["valor"] is not None else "--"
                baixa = "  << TAXA BAIXA" if s["pps"] < 35 else ""
                nome = "prob. movimento (arvore)" if s["arvore"] else "oscilacao"
                self.ax.set_ylabel(nome if s["arvore"] else "oscilacao (1.0 = repouso)")
                self.ax.set_title(f"pacotes/s: {s['pps']}{baixa} | {nome}: {val} | "
                                  f"limiar: {s['limiar']:.2f} | sensib: {s['sensib']:.2f}",
                                  fontsize=10, color="#d62728" if baixa else "black")
                h = np.array(s["hist"])
                if len(h):
                    x = -0.5 * np.arange(len(h) - 1, -1, -1)
                    self.linha.set_data(x, h)
                    self.ax.set_xlim(min(x[0], -30), 0)
                    top = 1.05 if s["arvore"] else max(s["limiar"] * 1.6, float(h.max()) * 1.1)
                    self.ax.set_ylim(0, top)
                self.lim.set_ydata([s["limiar"], s["limiar"]])
            self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Demo ao vivo de movimento por CSI.")
    p.add_argument("--serial", default=None, help="Porta serial (ex: COM13). Padrao: tenta detectar.")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--ip", default=None, help="IP do ESP32 (aparece no boot dele).")
    p.add_argument("--porta-udp", type=int, default=5555)
    p.add_argument("--intervalo-ms", type=float, default=20.0, help="Disparos UDP (20 = 50/s).")
    p.add_argument("--calibracao", type=float, default=60.0, help="Segundos de calibracao.")
    p.add_argument("--sensib", type=float, default=1.0, help="Maior = mais sensivel.")
    p.add_argument("--k", type=float, default=3.0, help="Limiar = normal + k * ruido.")
    p.add_argument("--n", type=int, default=10, help="Leituras na memoria da decisao (0,5 s cada).")
    p.add_argument("--m", type=int, default=4, help="Leituras acima do limiar para alertar.")
    p.add_argument("--modelo", default=None, help="Arvore treinada (modelo_arvore.joblib). Sem ela, usa limiar.")
    p.add_argument("--simular", action="store_true", help="Sinal sintetico, sem hardware.")
    p.add_argument("--replay", default=None, help="CSV de repouso (vazio) usado para calibrar.")
    p.add_argument("--alvo", default=None, help="CSV a reexibir depois da calibracao.")
    p.add_argument("--velocidade", type=float, default=1.0, help="Replay: 1 = tempo real, 0 = maximo.")
    p.add_argument("--texto", action="store_true", help="Sem janela grafica.")
    p.add_argument("--duracao", type=float, default=0.0, help="Encerra apos N segundos (0 = ate fechar).")
    return p.parse_args()


def main():
    args = parse_args()
    modo_hw = not args.simular and not args.replay
    if modo_hw and not args.ip:
        raise SystemExit("Informe --ip <IP do ESP32> (ou use --simular / --replay para ensaiar).")
    if args.calibracao < 10:
        raise SystemExit("--calibracao precisa ser de pelo menos 10 s (use 30 a 60 na demo).")

    parar, erros = threading.Event(), []
    contagem = {"invalidas": 0}
    modelo = None
    if args.modelo:
        import joblib
        pacote = joblib.load(args.modelo)
        if list(pacote["features"]) != list(cf.NOMES):
            raise SystemExit("Esse modelo foi treinado com outras medidas. Treine de novo com treinar_arvore.py.")
        modelo = pacote["modelo"]
        print(f"[modelo] arvore carregada de {args.modelo}")
    det = Detector(args.calibracao, args.sensib, args.k, args.n, args.m, modelo)
    stats = {"valores": 0, "alerta": 0, "episodios": 0, "ant": False}

    nome_log = datetime.datetime.now().strftime("demo_log_%Y-%m-%d_%H-%M-%S.csv")
    flog = open(nome_log, "w", newline="")
    w = csv.writer(flog)
    w.writerow(["hora", "oscilacao", "limiar", "pacotes_s", "alerta"])
    ui = None if args.texto else Janela(det, parar)
    fase_ant = ["calibrando"]
    msg_ant = [""]

    def processar(ts, amp):
        try:
            novo = det.add(ts, amp)
        except Exception as e:
            erros.append(f"Erro no detector: {e}")
            parar.set()
            return
        if det.fase != fase_ant[0]:
            fase_ant[0] = det.fase
            if det.fase == "ao_vivo":
                print(det.msg)
        elif det.fase == "calibrando" and det.msg and det.msg != msg_ant[0]:
            msg_ant[0] = det.msg
            print(det.msg)
        if novo:
            stats["valores"] += 1
            stats["alerta"] += int(det.alerta)
            stats["episodios"] += int(det.alerta and not stats["ant"])
            stats["ant"] = det.alerta
            w.writerow([datetime.datetime.now().isoformat(timespec="milliseconds"),
                        f"{det.valor:.4f}", f"{det.limiar:.4f}", det.pps, int(det.alerta)])
            if ui is None and not (args.replay and args.velocidade == 0):
                print(f"{'MOVIMENTO' if det.alerta else 'parado   '} | "
                      f"{'prob' if modelo is not None else 'osc'}={det.valor:.2f} lim={det.limiar:.2f} pps={det.pps}")

    def ao_fim():
        if ui is None:
            parar.set()

    if args.simular:
        threading.Thread(target=fonte_simulador, args=(processar, parar, args.calibracao), daemon=True).start()
        print("[simulacao] sinal sintetico a 50 pacotes/s")
    elif args.replay:
        alvo = args.alvo or args.replay
        print(f"[replay] calibrando com {args.replay}; reexibindo {alvo}")
        threading.Thread(target=fonte_replay, daemon=True,
                         args=(args.replay, alvo, args.calibracao, args.velocidade,
                               processar, parar, ao_fim)).start()
    else:
        porta = args.serial or achar_porta()
        print(f"Lendo {porta} a {args.baud} baud | disparando UDP para {args.ip}:{args.porta_udp}")
        threading.Thread(target=fonte_serial, daemon=True,
                         args=(porta, args.baud, processar, parar, erros, contagem)).start()
        threading.Thread(target=disparar_udp, daemon=True,
                         args=(args.ip, args.porta_udp, args.intervalo_ms / 1000.0, parar)).start()

    print(f"Calibrando por {args.calibracao:.0f} s. Mantenha a area do link parada.")
    inicio = time.time()
    ultimo_aviso = time.time()
    try:
        while not parar.is_set():
            if args.duracao and time.time() - inicio > args.duracao:
                break
            if ui is not None:
                if not ui.aberta():
                    break
                ui.tick()
            time.sleep(0.03)
            if modo_hw and det.total == 0 and time.time() - ultimo_aviso > 5:
                print("Sem dados ha 5 s (ESP32 ligado? IP certo? Serial Monitor fechado?)")
                ultimo_aviso = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        parar.set()
        flog.close()

    for e in erros:
        print(e)
    if stats["valores"]:
        print(f"\nResumo ao vivo: {stats['valores']} leituras | em alerta {stats['alerta'] / stats['valores']:.1%} "
              f"do tempo | {stats['episodios']} episodio(s) de alerta | pacotes invalidos: {contagem['invalidas']}")
    print(f"Log salvo em {nome_log}")


if __name__ == "__main__":
    main()
