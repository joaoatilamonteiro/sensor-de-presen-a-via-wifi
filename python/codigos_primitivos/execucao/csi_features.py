"""
Features CSI compartilhadas entre o TREINO (treinar_arvore.py) e a DEMO (demo_ao_vivo.py).
As duas usam exatamente este codigo, para o que a arvore ve ao vivo ser igual ao que ela viu no treino.

Ideia: cada janela de 1 s vira 5 numeros RELATIVOS ao repouso do dia (1.0 = "como parado").
Nao ha nenhuma feature que identifique a sessao (RSSI, formato absoluto do espectro etc.).
"""

import numpy as np

N_SUB = 64            # subportadoras do bloco estavel (LLTF)
MIN_PKTS = 20         # minimo de pacotes numa janela de 1 s
JANELA_MS = 1000
PASSO_MS = 500
NOMES = ["osc_media", "osc_p90", "decorr_1", "decorr_10", "energia"]


class Base:
    """O 'normal' do dia: aprendido em ~60 s com a area parada."""

    def __init__(self, mask, sd, d1, d10, en):
        self.mask, self.sd, self.d1, self.d10, self.en = mask, sd, d1, d10, en


def janelas(ts, passo=PASSO_MS):
    """Gera (i0, i1, t_fim) das janelas de 1 s com pacotes suficientes. ts precisa estar ordenado."""
    t = ts[0] + JANELA_MS
    while t <= ts[-1]:
        i0 = np.searchsorted(ts, t - JANELA_MS, side="right")
        i1 = np.searchsorted(ts, t, side="right")
        if i1 - i0 >= MIN_PKTS:
            yield i0, i1, t
        t += passo


def normalizar(amp, mask):
    """Tira subportadoras nulas e divide cada pacote pela propria amplitude media (cancela ganho)."""
    raw = amp[:, mask]
    x = raw / np.clip(raw.mean(axis=1, keepdims=True), 1e-9, None)
    return x, raw


def _medidas(x, raw):
    s = x.std(axis=0)                                           # oscilacao por subportadora
    xc = x - x.mean(axis=1, keepdims=True)
    u = xc / (np.linalg.norm(xc, axis=1, keepdims=True) + 1e-9)
    d1 = 1.0 - float((u[1:] * u[:-1]).sum(axis=1).mean())       # decorrelacao pacote a pacote
    d10 = 1.0 - float((u[10:] * u[:-10]).sum(axis=1).mean()) if len(u) > 10 else d1
    en = float(raw.mean(axis=1).std() / raw.mean())             # variacao de energia
    return s, d1, d10, en


def _features(base, s, d1, d10, en):
    z = s / base.sd
    return np.array([z.mean(), np.percentile(z, 90),
                     d1 / base.d1, d10 / base.d10, en / base.en])


def calibrar(ts, amp):
    """ts, amp (n x 64) de um trecho PARADO -> (Base, matriz de features do proprio repouso)."""
    mask = amp.mean(axis=0) > 0.2 * amp.mean()
    x, raw = normalizar(amp, mask)
    S, D1, D10, EN = [], [], [], []
    for i0, i1, _ in janelas(ts):
        s, d1, d10, en = _medidas(x[i0:i1], raw[i0:i1])
        S.append(s), D1.append(d1), D10.append(d10), EN.append(en)
    if len(S) < 15:
        return None, None
    S = np.stack(S)
    base = Base(mask, np.clip(S.mean(axis=0), 1e-6, None),
                max(float(np.median(D1)), 1e-6), max(float(np.median(D10)), 1e-6),
                max(float(np.median(EN)), 1e-6))
    F = np.array([_features(base, s, a, b, c) for s, a, b, c in zip(S, D1, D10, EN)])
    return base, F


def features_janela(base, x, raw):
    s, d1, d10, en = _medidas(x, raw)
    return _features(base, s, d1, d10, en)


def features_sessao(base, ts, amp, passo=PASSO_MS):
    """Todas as janelas de uma gravacao -> (tempos_ms, matriz de features)."""
    x, raw = normalizar(amp, base.mask)
    tempos, F = [], []
    for i0, i1, t in janelas(ts, passo):
        F.append(features_janela(base, x[i0:i1], raw[i0:i1]))
        tempos.append(t - ts[0])
    return np.array(tempos), np.array(F)
