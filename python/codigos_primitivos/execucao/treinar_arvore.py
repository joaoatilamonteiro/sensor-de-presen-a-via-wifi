"""
Treina uma ARVORE DE DECISAO que separa  vazio  x  movimento/pessoa  a partir das gravacoes de CSI.

Estrutura esperada (a mesma do seu zip):
    dados/vazio/csi_vazio_2026-10-17_16-33-53_8edca7e4.csv
    dados/sentado/...   dados/andando/...   (qualquer pasta que nao seja 'vazio' conta como pessoa)

Uso:
    python treinar_arvore.py --dados dados
    python treinar_arvore.py --dados dados --excluir ead31c83        (separa uma sessao para teste ao vivo)

O que ele faz:
    1. Le cada sessao e descarta as de taxa baixa (< --min-pps) ou curtas demais.
    2. Para cada DIA, usa a gravacao de vazio mais longa como "repouso" (primeiros --calib-s segundos)
       e mede todas as janelas de 1 s em unidades desse repouso (mesmo que a calibracao da demo).
    3. Valida de 3 jeitos (dentro do dia, entre dias, deixando uma sessao inteira de fora).
    4. Treina a arvore final, imprime as regras, salva modelo_arvore.joblib e arvore.png.

Requisitos: pip install numpy pandas scikit-learn joblib matplotlib
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

import csi_features as cf

warnings.filterwarnings("ignore")
TAMANHO = 384


# ----------------------------------------------------------------------------
# Leitura das sessoes
# ----------------------------------------------------------------------------
def ler_csv(caminho, cache_dir):
    """Devolve (ts_ms, amp[n x 64]) de um CSV; usa cache .npz para nao reler arquivos grandes."""
    os.makedirs(cache_dir, exist_ok=True)
    chave = f"{os.path.basename(caminho)}_{os.path.getsize(caminho)}.npz"
    destino = os.path.join(cache_dir, chave)
    if os.path.exists(destino):
        d = np.load(destino)
        return d["ts"], d["amp"]
    cols = [4, 6] + list(range(7, 7 + TAMANHO))
    tipos = {c: np.float32 for c in cols}
    tipos[4] = np.float64
    ts, amp = [], []
    for ch in pd.read_csv(caminho, header=None, skiprows=1, usecols=cols, dtype=tipos,
                          chunksize=40000, on_bad_lines="skip", low_memory=False):
        ch = ch.dropna()
        ch = ch[ch.iloc[:, 1] == TAMANHO]
        if ch.empty:
            continue
        v = ch.iloc[:, 2:].to_numpy(np.float32)
        amp.append(np.hypot(v[:, 0::2], v[:, 1::2])[:, :cf.N_SUB])
        ts.append(ch.iloc[:, 0].to_numpy())
    if not ts:
        return np.array([]), np.zeros((0, cf.N_SUB), np.float32)
    ts, amp = np.concatenate(ts), np.concatenate(amp)
    np.savez(destino, ts=ts, amp=amp)
    return ts, amp


def trecho_monotono(ts, amp):
    """Se o ESP32 reiniciou no meio da gravacao (ts volta a zero), fica o trecho mais longo."""
    quebras = np.flatnonzero(np.diff(ts) < 0) + 1
    if len(quebras) == 0:
        return ts, amp
    lim = np.r_[0, quebras, len(ts)]
    i = int(np.argmax(np.diff(lim)))
    return ts[lim[i]:lim[i + 1]], amp[lim[i]:lim[i + 1]]


def descobrir_sessoes(pasta, cache_dir, min_pps, excluir):
    sessoes = []
    for arq in sorted(glob.glob(os.path.join(pasta, "*", "*.csv"))):
        nome = os.path.basename(arq)
        if any(e in nome for e in excluir):
            print(f"  (excluida por --excluir) {nome}")
            continue
        rotulo = os.path.basename(os.path.dirname(arq)).lower()
        m = re.search(r"(\d{4}-\d{2}-\d{2})", nome)
        dia = m.group(1) if m else "sem-data"
        ts, amp = ler_csv(arq, cache_dir)
        if len(ts) < 2:
            continue
        ts, amp = trecho_monotono(ts, amp)
        dur = (ts[-1] - ts[0]) / 1000.0
        pps = len(ts) / dur if dur > 0 else 0.0
        motivo = ""
        if pps < min_pps:
            motivo = f"taxa baixa ({pps:.1f} pacotes/s)"
        elif dur < 20:
            motivo = f"curta demais ({dur:.0f} s)"
        sessoes.append(dict(nome=nome.replace(".csv", ""), rotulo=rotulo, dia=dia, ts=ts, amp=amp,
                            pps=pps, dur=dur, motivo=motivo))
    return sessoes


# ----------------------------------------------------------------------------
# Montagem do dataset
# ----------------------------------------------------------------------------
def montar_dataset(sessoes, calib_s, corte):
    linhas, avisos = [], []
    for dia in sorted({s["dia"] for s in sessoes}):
        doDia = [s for s in sessoes if s["dia"] == dia and not s["motivo"]]
        vazios = [s for s in doDia if s["rotulo"] == "vazio"]
        if not vazios:
            if doDia:
                avisos.append(f"{dia}: sem gravacao de vazio utilizavel -> {len(doDia)} sessao(oes) ficaram de fora "
                              f"(sem repouso do dia nao da para normalizar)")
            continue
        ref = max(vazios, key=lambda s: s["dur"])
        m = ref["ts"] < ref["ts"][0] + calib_s * 1000
        base, _ = cf.calibrar(ref["ts"][m], ref["amp"][m])
        if base is None:
            avisos.append(f"{dia}: calibracao falhou em {ref['nome']}")
            continue
        for s in doDia:
            t, F = cf.features_sessao(base, s["ts"], s["amp"], passo=1000)
            if len(t) == 0:
                continue
            keep = np.ones(len(t), bool)
            if s is ref:
                keep &= t >= (calib_s + 1) * 1000          # nao reaproveita a janela de calibracao
            if s["rotulo"] != "vazio" and s["dur"] > 10 * corte:
                keep &= (t >= corte * 1000) & (t <= (s["dur"] - corte) * 1000)
            for ti, fi in zip(t[keep], F[keep]):
                linhas.append(dict(sessao=s["nome"][:40], rotulo=s["rotulo"], dia=dia,
                                   bloco=int(ti // 120000), y=int(s["rotulo"] != "vazio"),
                                   **dict(zip(cf.NOMES, fi))))
    return pd.DataFrame(linhas), avisos


# ----------------------------------------------------------------------------
# Avaliacao
# ----------------------------------------------------------------------------
def nova_arvore(args):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=args.profundidade, min_samples_leaf=args.folha,
                                  class_weight="balanced", random_state=0)


def com_memoria(pred):
    """Mesma ideia da demo: alerta se >= 2 das ultimas 5 janelas (1 s) acusam movimento."""
    f = np.asarray(pred, int)
    return np.array([f[max(0, i - 4):i + 1].sum() >= 2 for i in range(len(f))])


def taxa(pred, y_real):
    return float(np.mean(pred)) if len(pred) else float("nan")


def avaliar(D, args):
    from sklearn.model_selection import GroupKFold
    X = cf.NOMES

    print("\n" + "=" * 72)
    print("A) DENTRO DO DIA (blocos de 2 min separados entre treino e teste)")
    print("=" * 72)
    for dia, Dd in D.groupby("dia"):
        if Dd.y.nunique() < 2:
            continue
        if (Dd.y == 1).sum() < 100 or (Dd.y == 0).sum() < 100:
            print(f"  {dia}: poucas janelas ({int((Dd.y == 1).sum())} de pessoa, "
                  f"{int((Dd.y == 0).sum())} de vazio) -> teste por blocos nao e confiavel, ignorado")
            continue
        g = Dd.sessao + "_" + Dd.bloco.astype(str)
        pred = np.zeros(len(Dd), int)
        for tr, te in GroupKFold(5).split(Dd, Dd.y, g):
            m = nova_arvore(args).fit(Dd.iloc[tr][X], Dd.y.iloc[tr])
            pred[te] = m.predict(Dd.iloc[te][X])
        Dd = Dd.assign(p=pred)
        for s, Ds in Dd.groupby("sessao"):
            nome = "falso alarme" if Ds.y.iloc[0] == 0 else "deteccao   "
            print(f"  {dia} {Ds.rotulo.iloc[0]:8s} {nome}: {taxa(Ds.p, None):6.1%}   ({s[:28]})")

    print("\n" + "=" * 72)
    print("B) ENTRE DIAS (treina num dia, testa em outro, cada dia com o seu repouso)")
    print("=" * 72)
    dias = [d for d, Dd in D.groupby("dia") if Dd.y.nunique() == 2 and (Dd.y == 1).sum() >= 20]
    feito = False
    for d_tr in dias:
        for d_te in D.dia.unique():
            Dte = D[D.dia == d_te]
            if d_te == d_tr or Dte.y.nunique() < 2:
                continue
            m = nova_arvore(args).fit(D[D.dia == d_tr][X], D[D.dia == d_tr].y)
            Dte = Dte.assign(p=m.predict(Dte[X]))
            n_pos = int((D[D.dia == d_tr].y == 1).sum())
            aviso = "   <- POUCOS dados de pessoa no treino, nao confie" if n_pos < 100 else ""
            print(f"  treino {d_tr} ({n_pos} janelas de pessoa) -> teste {d_te}{aviso}")
            for s, Ds in Dte.groupby("sessao"):
                nome = "falso alarme" if Ds.y.iloc[0] == 0 else "deteccao   "
                print(f"      {Ds.rotulo.iloc[0]:8s} {nome}: {taxa(Ds.p, None):6.1%}   ({s[:28]})")
            feito = True
    if not feito:
        print("  (precisa de pelo menos 2 dias, cada um com vazio e pessoa)")

    print("\n" + "=" * 72)
    print("C) DEIXANDO UMA SESSAO INTEIRA DE FORA  (o teste mais honesto)")
    print("   Em 'vazio', o resultado e o CONTROLE NEGATIVO: vazio gravado em outro momento.")
    print("=" * 72)
    print(f"  {'sessao':42s} {'por janela':>11s} {'com memoria':>12s}")
    for s in D.sessao.unique():
        tr, te = D[D.sessao != s], D[D.sessao == s]
        if tr.y.nunique() < 2:
            continue
        m = nova_arvore(args).fit(tr[X], tr.y)
        p = m.predict(te[X])
        rot = "FALSO ALARME" if te.y.iloc[0] == 0 else "deteccao"
        print(f"  {te.rotulo.iloc[0]:8s} {s[7:30]:22s} {rot:13s}{np.mean(p):6.1%}  {np.mean(com_memoria(p)):9.1%}")


# ----------------------------------------------------------------------------
def main():
    import joblib
    from sklearn.tree import export_text

    p = argparse.ArgumentParser(description="Treina a arvore de decisao do radar CSI.")
    p.add_argument("--dados", default="dados", help="Pasta com subpastas por classe (vazio, sentado, ...).")
    p.add_argument("--saida", default="modelo_arvore.joblib")
    p.add_argument("--calib-s", type=float, default=60.0, help="Segundos de repouso usados como base do dia.")
    p.add_argument("--min-pps", type=float, default=25.0, help="Descarta sessoes com menos pacotes/s.")
    p.add_argument("--profundidade", type=int, default=3)
    p.add_argument("--folha", type=int, default=50, help="Minimo de janelas por folha da arvore.")
    p.add_argument("--corte", type=float, default=5.0, help="Segundos cortados do inicio/fim de sessoes com pessoa.")
    p.add_argument("--excluir", nargs="*", default=[], help="Trechos de nome de arquivo para deixar de fora.")
    p.add_argument("--cache", default=".cache_csi")
    args = p.parse_args()

    print("Lendo sessoes...")
    sessoes = descobrir_sessoes(args.dados, args.cache, args.min_pps, args.excluir)
    print(f"\n{'classe':9s} {'dia':11s} {'min':>6s} {'pkt/s':>6s}  sessao")
    for s in sessoes:
        marca = "  <- DESCARTADA: " + s["motivo"] if s["motivo"] else ""
        print(f"{s['rotulo']:9s} {s['dia']:11s} {s['dur'] / 60:6.1f} {s['pps']:6.1f}  {s['nome'][-24:]}{marca}")

    D, avisos = montar_dataset(sessoes, args.calib_s, args.corte)
    for a in avisos:
        print("AVISO:", a)
    if D.empty or D.y.nunique() < 2:
        raise SystemExit("Dados insuficientes: preciso de vazio E pessoa, no mesmo dia, com taxa boa.")
    print(f"\nJanelas de 1 s: {len(D)} | vazio={int((D.y == 0).sum())} | pessoa={int((D.y == 1).sum())} "
          f"| sessoes={D.sessao.nunique()} | dias={D.dia.nunique()}")

    avaliar(D, args)

    modelo = nova_arvore(args).fit(D[cf.NOMES], D.y)
    print("\n" + "=" * 72)
    print("ARVORE FINAL (treinada com todas as sessoes)")
    print("=" * 72)
    print(export_text(modelo, feature_names=cf.NOMES))
    imp = sorted(zip(cf.NOMES, modelo.feature_importances_), key=lambda x: -x[1])
    print("Importancia das medidas:", ", ".join(f"{n}={v:.2f}" for n, v in imp if v > 0))
    print("(valores das medidas: 1.0 = repouso do dia; quanto maior, mais o sinal balanca)")

    joblib.dump({"modelo": modelo, "features": cf.NOMES, "versao": 1}, args.saida)
    print(f"\nModelo salvo em {args.saida}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        fig = plt.figure(figsize=(20, 8))
        plot_tree(modelo, feature_names=cf.NOMES, class_names=["vazio", "movimento"],
                  filled=True, rounded=True, fontsize=8, precision=2)
        fig.savefig("arvore.png", dpi=110, bbox_inches="tight")
        print("Desenho da arvore salvo em arvore.png")
    except Exception as e:
        print("Nao consegui desenhar a arvore:", e)


if __name__ == "__main__":
    main()
