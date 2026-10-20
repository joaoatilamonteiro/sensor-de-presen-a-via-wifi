#!/usr/bin/env python3
"""
analisar_sessoes.py — analisa 1 ou vários CSVs de CSI (cada CSV = uma sessão de coleta).

COM 1 CSV  -> análise da sessão: qualidade, perfil das subportadoras, mapa de calor,
              variabilidade no tempo e lista de "eventos" (momentos em que o sinal oscilou fora do normal).
COM 2+ CSV -> tudo acima + comparação entre sessões + teste de generalização
              "deixa uma sessão inteira de fora" (LOSO) + teste de sanidade (sessões da mesma condição).

Uso:
    python analisar_sessoes.py vazio1.csv
    python analisar_sessoes.py vazio1.csv vazio2.csv andando.csv sentado1.csv sentado2.csv
    python analisar_sessoes.py vazio=a.csv vazio=b.csv andando=c.csv sentado=d.csv sentado=e.csv
    python analisar_sessoes.py            # sem argumentos: pergunta os caminhos

O rótulo de cada sessão vem da coluna `rotulo` do CSV, ou de `rotulo=caminho` (que vence o CSV;
útil quando você rotulou errado). Condições listadas em --vazio (padrão: vazio) contam como "sala
vazia"; todas as outras contam como "tem pessoa".

Opções:  -o PASTA (padrão resultados_sessoes)   -j N pacotes por janela (padrão 10)
         --vazio vazio,sozinho                  --limiar-outlier Z (padrão 8)
Dependências: numpy pandas scipy scikit-learn matplotlib
"""
import argparse, itertools, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})
LOG = []
C_COND = {0: '#2a7fba', 1: '#d95f02'}
VAR_COLS = ['L_std_raw', 'L_std_norm', 'dabs', 'rng', 'rssi_std', 'amp_cv']


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.append(s)


# ------------------------------------------------------------------ leitura
def ler_csv(path):
    try:
        raw = pd.read_csv(path, header=None, skiprows=1, low_memory=False)
    except Exception as e:
        sys.exit(f'Erro lendo {path}: {e}')
    if raw.shape[1] < 9:
        sys.exit(f'{path}: formato inesperado ({raw.shape[1]} colunas).')
    meta = raw.iloc[:, :7].copy()
    meta.columns = ['sessao', 'rotulo', 'nota', 'ts_pc', 'ts_esp', 'rssi', 'tamanho']
    csi = raw.iloc[:, 7:].dropna(axis=1, how='all').to_numpy(float)
    rot = meta.rotulo.dropna().astype(str).mode()
    rotulo = rot.iloc[0] if len(rot) else os.path.splitext(os.path.basename(path))[0]
    meta['ts_pc'] = pd.to_datetime(meta.ts_pc, errors='coerce')
    meta['rssi'] = pd.to_numeric(meta.rssi, errors='coerce')
    return rotulo, meta.reset_index(drop=True), csi


def amplitudes(csi):
    n = csi.shape[1]
    amp = np.sqrt(csi[:, 0::2] ** 2 + csi[:, 1::2] ** 2)
    if n == 384:  # 3 blocos de 64 pares; 1º par de cada bloco = cabeçalho; centro nulo/DC
        iL = [i for i in range(2, 64) if not 28 <= i <= 36]
        iH = [64 + i for i in range(1, 64) if not 29 <= i <= 35]
        return amp[:, iL], amp[:, iH]
    return amp[:, (amp == 0).mean(0) < .95], None


def montar_sessao(rot, meta, csi, lim):
    XL, XH = amplitudes(csi)
    ts = meta.ts_pc
    t = (ts - ts.min()).dt.total_seconds().to_numpy() if ts.notna().all() else np.arange(len(meta), dtype=float)
    rssi = meta.rssi.fillna(meta.rssi.median()).to_numpy(float)
    aL = XL.mean(1)
    med = np.median(aL); mad = 1.4826 * np.median(np.abs(aL - med))
    z = np.abs(aL - med) / max(mad, 1e-9)
    keep = (z <= lim) & np.isfinite(aL)
    s = dict(rot=rot, n_orig=len(aL), n_rem=int((~keep).sum()), t=t[keep], rssi=rssi[keep], XL=XL[keep],
             XH=None if XH is None else XH[keep])
    s['aL'] = s['XL'].mean(1)
    s['XLn'] = s['XL'] / s['aL'][:, None]
    if s['XH'] is not None:
        s['ratio'] = s['XH'].mean(1) / s['aL']
    s['dt'] = np.diff(s['t'])
    return s


# ------------------------------------------------------------------ janelas
def janelas(s, W):
    rows, stat = [], []
    for k in range(len(s['aL']) // W):
        ix = slice(k * W, (k + 1) * W)
        f, r = s['XLn'][ix], s['XL'][ix]
        rows.append(dict(w=k, t0=s['t'][ix][0] / 60, t1=s['t'][ix][-1] / 60,
                         L_std_raw=r.std(0).mean(), L_std_norm=f.std(0).mean(),
                         dabs=np.abs(np.diff(f, axis=0)).mean(), rng=(f.max(0) - f.min(0)).mean(),
                         rssi_std=s['rssi'][ix].std(), amp_cv=r.mean(1).std() / r.mean(),
                         rssi_mean=s['rssi'][ix].mean()))
        stat.append(f.mean(0))
    return pd.DataFrame(rows), np.array(stat)


def achar_eventos(d, z_lim=4):
    x = d.L_std_norm.to_numpy()
    med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
    z = (x - med) / max(mad, 1e-9)
    flag = z > z_lim
    ev, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]: j += 1
            ev.append((d.t0.iloc[i], d.t1.iloc[j], float(z[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    return flag, ev


# ------------------------------------------------------------------ modelos
def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000, class_weight='balanced'))


def transf_var(D):
    return np.log(D[VAR_COLS].to_numpy() + 1e-2)


def loso(D, sets):
    nomes = list(D.sessao.unique()); rows = []; pooled = {k: ([], []) for k in sets}
    for h in nomes:
        te = (D.sessao == h).to_numpy(); tr = ~te
        yh = int(D.y[te].iloc[0])
        if D.y[tr].nunique() < 2:
            rows.append((h, yh, None, np.nan, np.nan)); continue
        for sn, F in sets.items():
            p = logreg().fit(F[tr], D.y[tr]).predict_proba(F[te])[:, 1]
            rows.append((h, yh, sn, float(np.mean((p > .5) == yh)), float(p.mean())))
            pooled[sn][0].extend([yh] * len(p)); pooled[sn][1].extend(p)
    return pd.DataFrame(rows, columns=['sessao', 'y', 'conjunto', 'acerto', 'p_medio']), pooled


def blocos(npos, K, purge):
    chunks = np.array_split(np.arange(npos), K)
    out = []
    for ch in chunks:
        lo, hi = ch.min() - purge, ch.max() + purge
        out.append((np.array([p for p in range(npos) if p < lo or p > hi]), ch))
    return out


def sanidade(D, sets, cond_de):
    nomes = list(D.sessao.unique()); res = []
    for a, b in itertools.combinations(nomes, 2):
        if cond_de[a] != cond_de[b]: continue
        ia, ib = np.where(D.sessao == a)[0], np.where(D.sessao == b)[0]
        K = min(5, min(len(ia), len(ib)) // 8)
        if K < 2: continue
        fa, fb = blocos(len(ia), K, 1), blocos(len(ib), K, 1)
        for sn, F in sets.items():
            ys, ps = [], []
            for k in range(K):
                tr = np.r_[ia[fa[k][0]], ib[fb[k][0]]]; te = np.r_[ia[fa[k][1]], ib[fb[k][1]]]
                ytr = np.r_[np.zeros(len(fa[k][0])), np.ones(len(fb[k][0]))]
                yte = np.r_[np.zeros(len(fa[k][1])), np.ones(len(fb[k][1]))]
                m = logreg().fit(F[tr], ytr); ps.extend(m.predict_proba(F[te])[:, 1]); ys.extend(yte)
            res.append((f'{a} × {b}', sn, roc_auc_score(ys, ps)))
    return pd.DataFrame(res, columns=['par', 'conjunto', 'auc'])


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser(description='Analisa 1 ou vários CSVs de CSI.')
    ap.add_argument('csvs', nargs='*', help='caminhos (ou rotulo=caminho)')
    ap.add_argument('-o', '--saida', default='resultados_sessoes')
    ap.add_argument('-j', '--janela', type=int, default=10)
    ap.add_argument('--vazio', default='vazio', help='nomes de condição que contam como sala vazia (separados por vírgula)')
    ap.add_argument('--limiar-outlier', type=float, default=8.0)
    a = ap.parse_args()
    itens = a.csvs
    if not itens:
        print('Digite um caminho de CSV por linha (use rotulo=caminho para forçar o rótulo). Linha vazia termina.')
        while True:
            x = input('> ').strip().strip('"')
            if not x: break
            itens.append(x)
    if not itens: sys.exit('Nenhum CSV informado.')
    os.makedirs(a.saida, exist_ok=True)
    P = lambda f: os.path.join(a.saida, f)
    vazio_nomes = {x.strip().lower() for x in a.vazio.split(',')}
    W = max(3, a.janela)

    # ---- carregar sessões
    brutas = []
    for it in itens:
        if '=' in it and not os.path.exists(it):
            rot, path = it.split('=', 1)
            _, meta, csi = ler_csv(path)
        else:
            rot, meta, csi = ler_csv(it)
        brutas.append((rot.strip(), meta, csi))
    cont = pd.Series([r for r, _, _ in brutas]).value_counts(); vistos = {}
    S = []
    for rot, meta, csi in brutas:
        vistos[rot] = vistos.get(rot, 0) + 1
        s = montar_sessao(rot, meta, csi, a.limiar_outlier)
        s['nome'] = rot if cont[rot] == 1 else f'{rot} ({vistos[rot]})'
        s['y'] = 0 if rot.lower() in vazio_nomes else 1
        S.append(s)
    N = len(S)
    cores = plt.cm.tab10(np.linspace(0, 1, 10))

    say('=' * 64); say(f'{N} sessão(ões) | janela = {W} pacotes | "vazio" = {sorted(vazio_nomes)}'); say('=' * 64)
    for s in S:
        d = s['dt']; dur = s['t'][-1] / 60
        s['taxa'] = len(s['t']) / max(s['t'][-1], 1)
        s['fraca'] = float((s['ratio'] < .85).mean()) if s['XH'] is not None else np.nan
        say(f"[{s['nome']}] {'VAZIO' if s['y']==0 else 'PESSOA'} | {len(s['t'])} pacotes ({s['n_rem']} outliers removidos) | {dur:.1f} min | "
            f"{s['taxa']:.2f} pac/s | RSSI {s['rssi'].mean():.1f} ± {s['rssi'].std():.2f} dBm"
            + (f" | HT fraco {100*s['fraca']:.0f}%" if s['XH'] is not None else ''))

    # ---- janelas e eventos
    DF, STAT = [], []
    for s in S:
        d, st = janelas(s, W)
        d['sessao'] = s['nome']; d['y'] = s['y']
        flag, ev = achar_eventos(d); d['evento'] = flag; s['ev'] = ev; s['d'] = d
        DF.append(d); STAT.append(st)
    D = pd.concat(DF, ignore_index=True); STAT = np.vstack(STAT)
    say('\n--- Eventos (janelas em que a oscilação do sinal fica muito acima do normal da própria sessão)')
    for s in S:
        if s['ev']:
            say(f"[{s['nome']}] " + '; '.join(f'{x0:.1f}–{x1:.1f} min (z={z:.0f})' for x0, x1, z in s['ev']))
        else:
            say(f"[{s['nome']}] nenhum evento")
    say('\n--- Variabilidade por sessão (mediana das janelas)')
    tab = D.groupby('sessao', sort=False)[VAR_COLS].median().round(3)
    say(tab.to_string())

    # ================= figuras (valem para 1 ou N sessões)
    nomes = [s['nome'] for s in S]
    # fig1 qualidade
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].boxplot([s['rssi'] for s in S], tick_labels=nomes, showfliers=False)
    ax[0, 0].set(title='RSSI por sessão', ylabel='dBm')
    for i, s in enumerate(S):
        ax[0, 1].plot(s['t'] / 60, pd.Series(s['aL']).rolling(30, center=True).mean(), color=cores[i % 10], label=s['nome'])
    ax[0, 1].set(title='Amplitude média L-LTF (média móvel de 30 pacotes)', xlabel='minutos'); ax[0, 1].legend(fontsize=7)
    ax[1, 0].bar(nomes, [100 * s['fraca'] if s['XH'] is not None else 0 for s in S], color=[C_COND[s['y']] for s in S])
    ax[1, 0].set(title='% de pacotes com HT-LTF "fraco" (razão H/L < 0.85)', ylabel='%')
    ax[1, 1].bar(nomes, [s['taxa'] for s in S], color=[C_COND[s['y']] for s in S])
    for i, s in enumerate(S): ax[1, 1].text(i, s['taxa'], f"{s['n_rem']} out.", ha='center', va='bottom', fontsize=7)
    ax[1, 1].set(title='Pacotes por segundo (texto = outliers removidos)', ylabel='pac/s')
    for x in (ax[0, 0], ax[1, 0], ax[1, 1]): x.tick_params(axis='x', rotation=25)
    plt.tight_layout(); plt.savefig(P('fig1_qualidade.png')); plt.close()
    # fig2 perfil
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, s in enumerate(S):
        ax[0].plot(s['XL'].mean(0), color=cores[i % 10], label=s['nome'])
        ax[1].plot(s['XL'].std(0), color=cores[i % 10], label=s['nome'])
    ax[0].set(title='Amplitude média por subportadora (L-LTF)', ylabel='amplitude')
    ax[1].set(title='Desvio-padrão temporal por subportadora (L-LTF)', xlabel='subportadora válida', ylabel='dp')
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(P('fig2_perfil_subportadoras.png')); plt.close()
    # fig3 heatmaps
    nc = min(3, N); nr = int(np.ceil(N / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(5 * nc, 3.6 * nr), squeeze=False)
    mats = []
    for s in S:
        X = np.hstack([s['XL'], s['XH']]) if s['XH'] is not None else s['XL']
        mats.append(X / X.mean(1, keepdims=True))
    vmin, vmax = np.percentile(np.vstack(mats), [2, 98])
    for i, s in enumerate(S):
        x = ax[i // nc, i % nc]
        im = x.imshow(mats[i].T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=[0, s['t'][-1] / 60, mats[i].shape[1], 0])
        x.set(title=s['nome'], xlabel='minutos'); x.grid(False)
    for j in range(N, nr * nc): ax[j // nc, j % nc].axis('off')
    plt.tight_layout(); plt.savefig(P('fig3_heatmaps.png')); plt.close()
    # fig4 variabilidade por sessão
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for x, col in zip(ax, ['L_std_raw', 'L_std_norm', 'rssi_std', 'dabs']):
        bp = x.boxplot([D[D.sessao == n][col] for n in nomes], tick_labels=nomes, patch_artist=True, showfliers=False)
        for patch, s in zip(bp['boxes'], S): patch.set_facecolor(C_COND[s['y']]); patch.set_alpha(.6)
        x.set(title=f'{col} (por janela de {W})'); x.tick_params(axis='x', rotation=30)
    plt.suptitle('Variabilidade por sessão (azul = vazio, laranja = pessoa)', y=1.02)
    plt.tight_layout(); plt.savefig(P('fig4_variabilidade_por_sessao.png'), bbox_inches='tight'); plt.close()
    # fig5 eventos no tempo
    fig, ax = plt.subplots(N, 1, figsize=(11, 2.4 * N), squeeze=False)
    for i, s in enumerate(S):
        x = ax[i, 0]; d = s['d']
        x.plot((d.t0 + d.t1) / 2, d.L_std_norm, color=C_COND[s['y']], lw=1)
        for x0, x1, z in s['ev']: x.axvspan(x0, x1, color='red', alpha=.2)
        x.set(ylabel=s['nome'], title='' if i else 'Oscilação do sinal (L normalizada) por janela; faixa vermelha = evento')
    ax[-1, 0].set_xlabel('minutos'); plt.tight_layout(); plt.savefig(P('fig5_eventos_no_tempo.png')); plt.close()

    # ================= só com 2+ sessões
    if N >= 2:
        # PCA estático
        Z = PCA(2).fit_transform(StandardScaler().fit_transform(STAT))
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for i, s in enumerate(S):
            m = (D.sessao == s['nome']).to_numpy()
            ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=.6, color=cores[i % 10], marker='o' if s['y'] == 0 else '^', label=f"{s['nome']} ({'vazio' if s['y']==0 else 'pessoa'})")
        ax.set(title='PCA do perfil estático (L normalizada, média por janela)\nse os grupos são por SESSÃO e não por condição, o modelo aprende a sessão', xlabel='PC1', ylabel='PC2')
        ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(P('fig6_pca_por_sessao.png')); plt.close()

        sets = {'RSSI médio': D[['rssi_mean']].to_numpy(), 'Perfil estático (L)': STAT, 'Variabilidade': transf_var(D)}
        cond_de = {s['nome']: s['y'] for s in S}
        # LOSO
        say('\n' + '=' * 64); say('TESTE "DEIXA UMA SESSÃO DE FORA" (treina nas outras, testa na sessão inteira nunca vista)'); say('=' * 64)
        L, pooled = loso(D, sets)
        rodou = L.conjunto.notna().any()
        if not rodou:
            say('Não foi possível: ao tirar qualquer sessão, as restantes não têm as duas condições (vazio e pessoa).\n'
                'Precisa de pelo menos 2 sessões de vazio e 2 com pessoa (ou 1+1 além da testada).')
        else:
            tbl = L.dropna(subset=['conjunto']).pivot(index='sessao', columns='conjunto', values='acerto').reindex(nomes)
            say('% de janelas classificadas corretamente em cada sessão de fora:')
            say((100 * tbl).round(0).to_string())
            pm = L.dropna(subset=['conjunto']).pivot(index='sessao', columns='conjunto', values='p_medio').reindex(nomes)
            say('\nP(pessoa) média atribuída a cada sessão de fora (vazio deveria dar ~0, pessoa ~1):')
            say(pm.round(2).to_string())
            ycond = pd.Series({s['nome']: s['y'] for s in S})
            bal = {}
            for sn in sets:
                g = L[(L.conjunto == sn)].groupby('y').acerto.mean()
                bal[sn] = float(g.mean()) if len(g) else np.nan
            say('\nAcerto balanceado (média entre vazio e pessoa): ' + ' | '.join(f'{k}: {v:.2f}' for k, v in bal.items()))
            for sn, (yy, pp) in pooled.items():
                if len(set(yy)) == 2: say(f'AUC agregada [{sn}]: {roc_auc_score(yy, pp):.3f}')
        # sanidade
        say('\n' + '=' * 64); say('TESTE DE SANIDADE (sessões da MESMA condição não deveriam se separar: AUC ~0.5)'); say('=' * 64)
        SA = sanidade(D, sets, cond_de)
        if SA.empty:
            say('Não há duas sessões da mesma condição (ou janelas insuficientes).')
        else:
            say(SA.pivot(index='par', columns='conjunto', values='auc').round(3).to_string())
        # fig7
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        if rodou:
            tb = tbl.fillna(0); w = .8 / len(sets); xs = np.arange(len(tb))
            for k, sn in enumerate(sets):
                ax[0].bar(xs + k * w, 100 * tb[sn].values, w, label=sn)
            ax[0].set_xticks(xs + .4 - w / 2); ax[0].set_xticklabels(tb.index, rotation=25)
            ax[0].axhline(50, color='k', ls=':', lw=.8)
            ax[0].set(title='Sessão de fora: % de janelas acertadas', ylabel='%', ylim=(0, 105)); ax[0].legend(fontsize=7)
        else:
            ax[0].axis('off'); ax[0].text(.5, .5, 'teste não executado\n(faltam condições no treino)', ha='center')
        if not SA.empty:
            pv = SA.pivot(index='par', columns='conjunto', values='auc'); w = .8 / len(sets); xs = np.arange(len(pv))
            for k, sn in enumerate(sets): ax[1].bar(xs + k * w, pv[sn].values, w, label=sn)
            ax[1].set_xticks(xs + .4 - w / 2); ax[1].set_xticklabels(pv.index, rotation=25, fontsize=7)
            ax[1].axhline(.5, color='k', ls=':', lw=.8)
            ax[1].set(title='Sanidade: separar duas sessões da MESMA condição (queremos ~0.5)', ylabel='AUC', ylim=(0, 1.05)); ax[1].legend(fontsize=7)
        else:
            ax[1].axis('off'); ax[1].text(.5, .5, 'sem pares da mesma condição', ha='center')
        plt.tight_layout(); plt.savefig(P('fig7_teste_loso_e_sanidade.png')); plt.close()

        # leitura automática
        say('\n' + '=' * 64); say('LEITURA AUTOMÁTICA'); say('=' * 64)
        if rodou:
            e, v = bal.get('Perfil estático (L)', np.nan), bal.get('Variabilidade', np.nan)
            say(f'* Perfil estático em sessão nunca vista: acerto balanceado {e:.2f}. ' + ('Generaliza.' if e >= .85 else 'NÃO generaliza bem (provável efeito de sessão).'))
            say(f'* Variabilidade em sessão nunca vista: acerto balanceado {v:.2f}. ' +
                ('Promissor, mas confirme com mais sessões.' if v >= .85 else 'Generalização parcial.' if v >= .65 else 'NÃO generaliza.'))
        if not SA.empty:
            sv = SA[SA.conjunto == 'Variabilidade'].auc.mean(); ss = SA[SA.conjunto == 'Perfil estático (L)'].auc.mean()
            say(f'* Sanidade (mesma condição): perfil estático separa sessões com AUC {ss:.2f}; variabilidade com AUC {sv:.2f}.'
                + (' Se estiver perto de 1, o atributo denuncia a sessão e não deve ser usado.' if max(ss, sv) > .8 else ''))
        if np.mean([s['taxa'] for s in S]) < 10:
            say('* Taxa média < 10 pacotes/s: baixa para detectar movimento; busque 50-100 pacotes/s.')
        say('* Com poucas sessões por condição, trate qualquer resultado como indício, não como prova.')
    else:
        say('\n(1 sessão: comparação e teste de generalização exigem 2 ou mais CSVs.)')

    with open(P('resumo.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))
    print(f'\nArquivos salvos em: {os.path.abspath(a.saida)}')


if __name__ == '__main__':
    main()