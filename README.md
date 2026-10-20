# Radar de Wi-Fi (CSI) com ESP32

Detecção de **movimento de pessoas** perto do link Wi-Fi entre um ESP32 e o roteador, sem câmeras, usando o *Channel State Information* (CSI): números que o chip Wi-Fi mede em cada quadro recebido e que mudam quando algo se move no caminho do sinal.

- **Firmware (C++ / PlatformIO):** o ESP32 conecta ao Wi-Fi, mede o CSI de cada quadro e imprime uma linha por pacote na serial.
- **Python:** captura as linhas pela USB, calcula 5 medidas por janela de 1 s, treina uma **árvore de decisão** (`scikit-learn`) e roda uma demonstração ao vivo com gráfico em tempo real.
- **Classes:** `0` = vazio (ninguém no quarto) e `1` = pessoa/movimento (sentado, andando etc.). O rótulo é o **nome da pasta** de cada gravação.

> O sistema detecta **movimento**, não presença. Uma pessoa sentada e quieta quase não altera o canal e é pouco detectável.

```
PC ──UDP a cada 20 ms──► roteador ──► ESP32 ──USB (serial)──► PC
                                      mede o CSI              lê, calcula as 5 medidas
                                      de cada quadro          e a árvore decide: movimento?
```

## Índice

- [Requisitos](#requisitos)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Uso](#uso)
- [Formato dos dados](#formato-dos-dados)
- [Dados e modelos fora do Git](#dados-e-modelos-fora-do-git)
- [Problemas comuns](#problemas-comuns)
- [Scripts de apoio](#scripts-de-apoio)
- [Limitações](#limitações)

---

## Requisitos

### Hardware (só para gravar dados e usar a demo ao vivo)

| Item | Observação |
|---|---|
| Placa **ESP32** (`esp32dev`) | Precisa suportar CSI (ESP32 clássico serve). |
| Cabo USB **de dados** | Cabo só de carga não cria a porta serial. |
| Driver USB-serial | CP210x ou CH340, conforme a placa. |
| Roteador Wi-Fi **2.4 GHz** | O ESP32 clássico não conecta em 5 GHz. |
| PC | Na **mesma rede** do ESP32 (para a demo ao vivo). |

### Software

| Item | Para quê |
|---|---|
| [PlatformIO](https://platformio.org/) (extensão do VS Code ou CLI) | Compilar e gravar o firmware. |
| Python **3.12** (desenvolvido com 3.12.10; 3.10+ deve funcionar; o Colab usa 3.13) | Captura, treino e demo. |
| Dependências Python | Listadas em [`python/requirements.txt`](python/requirements.txt). |
| Jupyter (VS Code, JupyterLab ou Notebook) ou **Google Colab** | O fluxo principal está em `python/radar_csi_arvore.ipynb`. |

---

## Estrutura do repositório

```
.
├── platformio.ini                  # projeto PlatformIO: placa esp32dev, Arduino, monitor a 921600
├── src/                            # firmware do ESP32
│   ├── main.cpp                    #   conecta ao Wi-Fi e chama csiSetup()/csiLoop()
│   ├── csi_radar.cpp               #   ativa o CSI e imprime 1 linha por pacote na serial
│   └── csi_radar_secrets.h         #   SSID e senha do Wi-Fi (valores de exemplo, ver Instalação)
├── include/  lib/  test/           # pastas padrão do PlatformIO
├── python/
│   ├── requirements.txt            # dependências Python
│   ├── radar_csi_arvore.ipynb      # notebook principal: medidas, treino e demo ao vivo
│   ├── dados/                      # gravações de CSI, uma subpasta por classe (não versionado)
│   │   └── vazio/  sentado/  andando/  dormindo/
│   ├── modelos/                    # modelos treinados, .joblib (não versionado)
│   ├── foto_arvores_decisao/       # desenho das árvores, .png (não versionado)
│   └── codigos_primitivos/         # scripts de apoio e versões antigas (ver Scripts de apoio)
│       ├── codigos_captura/        #   escuta_pc.py, roteador_onda.py, analisar_csi.py
│       ├── execucao/               #   csi_features.py, treinar_arvore.py, demo_ao_vivo.py
│       └── analises_antigas/       #   figuras e resumo.txt das primeiras análises
├── .gitignore
└── README.md
```

Ao executar o notebook também são criados `python/demo_log/` (logs das demos) e `python/.cache_csi/` (cache de leitura dos CSVs). Os dois ficam fora do Git.

---

## Instalação

### Firmware (ESP32)

1. Edite o arquivo `src/csi_radar_secrets.h`, que vem com valores de exemplo, e coloque as credenciais da **sua** rede 2.4 GHz:

   ```cpp
   #ifndef CSI_RADAR_SECRETS_H
   #define CSI_RADAR_SECRETS_H
   #include <Arduino.h>

   const char* const WIFI_SSID = "nome_da_rede_2.4G";
   const char* const WIFI_PASSWORD = "senha_da_rede";

   #endif
   ```

   > Depois de colocar a senha real, evite fazer commit dela. `git update-index --skip-worktree src/csi_radar_secrets.h` faz o Git ignorar as suas alterações locais nesse arquivo.

2. Conecte o ESP32 por USB e, na raiz do repositório, grave o firmware:

   ```bash
   pio run -t upload
   ```

3. Abra o **Serial Monitor** a **921600 baud**:

   ```bash
   pio device monitor
   ```

   Devem aparecer `WiFi conectado!`, o **IP** do ESP32 e `CSI ativado`, seguido do cabeçalho `timestamp_ms,rssi,tamanho,csi_raw` e das linhas de CSI.

4. Anote o IP e **feche o Serial Monitor**: a porta serial só abre em um programa por vez.

### Python (execução local)

```bash
cd python
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

No VS Code, selecione `.venv` como kernel do notebook. A primeira célula de código do notebook também executa `%pip install`, então ela pode ser pulada se o passo acima já foi feito.

---

## Uso

> Os comandos do notebook (`%treinar_arvore`, `%demo_ao_vivo`) usam caminhos **relativos**. Abra o notebook com a pasta de trabalho em `python/`, onde ficam `dados/` e `modelos/`. No Colab isso já é o padrão.

O que cada parte do notebook `python/radar_csi_arvore.ipynb` mostra:

| Seção | O que mostra |
|---|---|
| 3 | As 5 medidas do sinal, testadas com um sinal sintético |
| 4 | Treino e validação da árvore de decisão com as gravações reais |
| 5.4 | Demonstração com sinal simulado (sem dados e sem hardware) |
| 5.5 a 5.7 | Replays das gravações reais, com limiar e com a árvore treinada |
| 5.8 | Execução real com o ESP32 (precisa de hardware) |
| 6 | Limitações e conclusões |

### Gravar dados (captura)

Cada sessão vira um CSV. Rode o script de captura dentro da pasta da classe, para o arquivo já cair no lugar certo:

```bash
cd python/dados/vazio
python ../../codigos_primitivos/codigos_captura/escuta_pc.py --rotulo vazio --nota "quarto vazio" --serial COM13 --baud 921600
```

Pontos importantes:

- **Use `--baud 921600`.** O padrão do script é 115200, que não bate com o firmware.
- `--serial` pode ser omitido se houver só uma porta serial no PC (Linux: algo como `/dev/ttyUSB0`).
- O `--rotulo` deve ser igual ao nome da pasta (`vazio`, `sentado`, `andando`...). O arquivo sai como `csi_<rotulo>_<data>_<id>.csv`.
- Encerre com `Ctrl+C`. O script mostra quantas linhas foram gravadas e quantas descartadas.
- Grave a **~50 pacotes/s**. Sessões com menos de 25 pacotes/s ou menos de 20 s são descartadas no treino.
- Grave **vazio e pessoa no mesmo dia**, pois o treino normaliza cada dia pelo seu próprio repouso.
- Opcional: `roteador_onda.py` (em `codigos_captura/`) envia 50 UDP/s ao ESP32 para gerar tráfego durante a captura. Edite o `IP_ESP32` dentro do arquivo antes de usar. O firmware já pinga o roteador a cada 20 ms por conta própria.

### Treinar a árvore

No notebook, execute as células em ordem até a seção 4 e rode:

```
%treinar_arvore --dados dados
```

O comando lê as sessões, monta as janelas de 1 s, valida de três jeitos (dentro do dia, entre dias, sessão inteira de fora), treina a árvore final e salva:

- `python/modelos/modelo_arvore.joblib`
- `python/foto_arvores_decisao/modelo_arvore.png`
- `python/.cache_csi/` (cache dos CSVs lidos; a primeira leitura é lenta)

Argumentos principais:

| Argumento | Padrão | Para que serve |
|---|---|---|
| `--dados` | `dados` | pasta com as subpastas por classe |
| `--saida` | `modelo_arvore.joblib` | nome do modelo |
| `--calib-s` | `60` | segundos de repouso usados como base do dia |
| `--min-pps` | `25` | descarta sessões com menos pacotes por segundo |
| `--profundidade` | `3` | profundidade máxima da árvore |
| `--folha` | `50` | mínimo de janelas por folha |
| `--corte` | `5` | segundos cortados do início e do fim das sessões com pessoa |
| `--excluir` | nenhum | trechos de nome de arquivo a deixar de fora (teste com sessão nunca vista) |
| `--sem-avaliacao` | desligado | pula a validação |

### Demonstração sem hardware

Não precisa de ESP32 nem de dados:

```
%demo_ao_vivo --simular --calibracao 10 --duracao 48
```

Com as suas gravações (replay: calibra com o início de uma gravação de vazio e reexibe outra):

```
%demo_ao_vivo --replay dados/vazio/<arquivo_vazio>.csv --alvo dados/andando/<arquivo_andando>.csv --velocidade 0 --texto
```

Por que dois arquivos: a demo tem duas fases. O `--replay` é o CSV de repouso (um vazio): os primeiros 60 s dele alimentam a **calibração**, sem espera. O `--alvo` é a gravação que você quer testar: depois da calibração, os pacotes dele chegam ao detector como se fossem ao vivo. Se omitir `--alvo`, o mesmo arquivo serve para as duas fases (mede o falso alarme do vazio contra ele mesmo). Para um teste justo, calibre e teste com gravações **do mesmo dia**.

`--velocidade` controla só o ritmo do alvo: `1` é tempo real, `6` é seis vezes mais rápido (a caminhada de ~6 min leva ~1 min), e `0` é o mais rápido possível, sem gráfico ao vivo (use com `--texto`).

Para testar com um modelo treinado, acrescente `--modelo modelo_arvore.joblib` (o arquivo é procurado em `modelos/`). O `modelo_arvore.joblib` já treinou com a gravação de caminhada. Para um teste honesto, treine um modelo que **nunca viu** essa gravação (seção 5.6 do notebook) e use-o no replay:

```
%treinar_arvore --dados dados --excluir ead31c83 --sem-avaliacao --saida modelo_sem_andando.joblib
```

(`ead31c83` é o identificador da gravação de caminhada de 16/10.)

### Demonstração ao vivo (com o ESP32)

Antes de rodar:

- o ESP32 está ligado na USB e conectado ao Wi-Fi;
- o PC está na mesma rede;
- o Serial Monitor do PlatformIO está **fechado**;
- a área entre o ESP32 e o roteador ficará **parada** durante a calibração.

Com limiar (não precisa de modelo):

```
%demo_ao_vivo --ip 192.168.1.165 --serial COM13 --calibracao 60
```

Com a árvore treinada:

```
%demo_ao_vivo --ip 192.168.1.165 --serial COM13 --calibracao 60 --modelo modelo_arvore.joblib
```

Troque `192.168.1.165` pelo IP que o seu ESP32 mostrou no boot e `COM13` pela sua porta. Opções úteis: `--sensib` (maior = mais sensível), `--k`, `--n`, `--m` (regra do alerta: *m* das últimas *n* leituras acima do limiar) e `--texto` (sem gráfico). O título do gráfico mostra **SEM MOVIMENTO**, **MOVIMENTO DETECTADO** ou **CALIBRANDO**, e aparece **TAXA BAIXA** se chegarem menos de 35 pacotes/s (o ideal é ~50).

**Calibre sempre no local, minutos antes de usar.** O "normal" do sinal muda de um dia para o outro; calibrar com o repouso de outro dia gera falso alarme.

### Rodar no Google Colab

Dá para ver o treino e os replays sem ter o ESP32.

1. Abra `python/radar_csi_arvore.ipynb` no Colab: *Arquivo → Abrir notebook → GitHub*, ou envie o arquivo.
2. Rode as células em ordem. A célula **2.1** baixa as gravações do Google Drive para `dados/<classe>/`: um link de arquivo por CSV, com `gdown --fuzzy`, sem zip. Os links ficam no dicionário `ARQUIVOS_DRIVE`; para incluir uma gravação nova, acrescente `"nome.csv": "ID_DO_ARQUIVO"` na classe certa (o ID é o trecho entre `/file/d/` e `/view` do link).
3. A célula pode ser rodada de novo: arquivos já baixados são pulados e um arquivo que falha não derruba os outros. Os arquivos precisam estar compartilhados como **"Qualquer pessoa com o link"**. Alguns CSVs têm centenas de MB, então o download e a primeira leitura no treino podem levar vários minutos.
4. Se algum falhar (`Cannot retrieve the public link of the file`), o motivo mais provável é a cota de downloads do Drive. Espere algumas horas e rode de novo, ou baixe no navegador e arraste o arquivo para `dados/<classe>/` no painel de arquivos do Colab.
5. Execute até a **seção 5.7** (replays). **Pare antes da 5.8:** ela usa o ESP32 na USB e a rede local, que o Colab não enxerga.

O que **não** funciona no Colab: a captura (`escuta_pc.py`) e a demo ao vivo com `--ip`/`--serial`, porque o Colab roda na nuvem e não enxerga a porta USB nem a sua rede local. O Colab também apaga `modelos/`, `demo_log/` e `.cache_csi/` do ambiente ao encerrar a sessão: baixe o `.joblib` que quiser guardar.

### Como ler o gráfico da demonstração

- **Título:** mostra o estado **neste instante** (borda direita do gráfico). Fica vermelho (**MOVIMENTO DETECTADO**) quando pelo menos 4 das últimas 10 leituras passam do limiar; verde é **SEM MOVIMENTO** e laranja é **CALIBRANDO**.
- **Linha abaixo do título:** `pacotes/s` (ideal ~50; abaixo de 35 aparece **TAXA BAIXA**), `valor` (última leitura), `limiar` (a linha vermelha) e `sensib`.
- **Eixo horizontal:** segundos atrás (0 = agora; guarda os últimos 120 s, uma leitura a cada 0,5 s).
- **Eixo vertical, sem modelo:** oscilação em unidades do repouso (1,0 = "como parado").
- **Eixo vertical, com árvore:** probabilidade de movimento (0 a 1), limiar em 0,5. Como a árvore é rasa, a curva fica perto de 0 ou de 1.
- **Cuidado ao interpretar um print:** o título reflete só o último instante. Se o movimento acabou antes do print, o título fica verde, mas os picos à esquerda são os movimentos dos 2 minutos anteriores.

---

## Formato dos dados

Linha impressa pelo ESP32:

```
timestamp_ms,rssi,tamanho,v0,v1,...,v383
```

O CSV gravado pelo `escuta_pc.py` acrescenta metadados à frente:

```
sessao_id,rotulo,nota,timestamp_pc,timestamp_ms_esp32,rssi,tamanho,csi_raw...
```

Cada pacote traz 384 valores (192 pares real/imaginário). O processamento usa a amplitude `hypot(real, imaginário)` das 64 primeiras subportadoras (bloco estável, LLTF) e calcula 5 medidas por janela de 1 s (passo de 0,5 s), todas relativas ao repouso do dia (1,0 = "como parado"): `osc_media`, `osc_p90`, `decorr_1`, `decorr_10` e `energia`. Os detalhes estão no notebook, seção 3.

---

## Dados e modelos fora do Git

O `.gitignore` deixa de fora o que é grande ou gerado:

| O que | Onde | Por quê |
|---|---|---|
| Gravações de CSI | `python/dados/**/*.csv` | Alguns arquivos passam de 100 MB (limite de um arquivo no GitHub). |
| Modelos treinados | `python/modelos/*.joblib` | São gerados pelo treino. |
| Desenhos das árvores | `python/foto_arvores_decisao/` | São gerados pelo treino. |
| Cache e logs | `python/.cache_csi/`, `python/demo_log/` | São gerados pelo notebook. |

Depois de clonar o repositório, para ter dados e modelos:

- **Dados:** grave os seus (seção [Gravar dados](#gravar-dados-captura)) ou use a célula 2.1 do notebook no Colab, que baixa as gravações de exemplo.
- **Modelos:** rode `%treinar_arvore --dados dados` no notebook.

---

## Problemas comuns

| Sintoma | O que verificar |
|---|---|
| Nenhuma porta serial encontrada | Cabo USB de dados, driver CP210x/CH340, e se outro programa (Serial Monitor) não está usando a porta. |
| CSV com só o cabeçalho / muitas linhas descartadas | Baud diferente de 921600, ou o ESP32 ainda não conectou ao Wi-Fi. |
| ESP32 não conecta | A rede precisa ser 2.4 GHz; confira SSID e senha em `src/csi_radar_secrets.h`. |
| **TAXA BAIXA** na demo | Menos de 35 pacotes/s: aproxime o ESP32 do roteador, confira o cabo e feche outros programas na serial. |
| Falso alarme com tudo parado | Recalibre no local; use `--sensib 0.8`; ou rode sem `--modelo`. |
| Download do Drive falha no Colab | Compartilhamento como "Qualquer pessoa com o link" e cota de downloads do Drive; rode a célula 2.1 de novo mais tarde. |
| Treino diz "dados insuficientes" | É preciso vazio **e** pessoa no mesmo dia, com taxa boa (≥ 25 pacotes/s). |

---

## Scripts de apoio

Ficam em `python/codigos_primitivos/`:

- `codigos_captura/escuta_pc.py`: captura das gravações (**ainda necessário** para gravar dados novos).
- `codigos_captura/roteador_onda.py`: gera tráfego UDP para o ESP32 durante a captura.
- `codigos_captura/analisar_csi.py`: análise exploratória com figuras. Usa `scipy`.
- `execucao/`: versão em scripts do que hoje está no notebook (`csi_features.py`, `treinar_arvore.py`, `demo_ao_vivo.py`). Rode de dentro dessa pasta, pois `treinar_arvore.py` e `demo_ao_vivo.py` importam `csi_features.py`. Pode estar defasada em relação ao notebook.
- `analises_antigas/`: figuras e `resumo.txt` de análises feitas sobre as primeiras gravações (~1 pacote/s).

---

## Limitações

- Com poucos dias de gravação, a árvore acaba sendo praticamente um corte em `osc_p90`. Mais sessões de vazio, a ~50 pacotes/s e em mais dias, melhoram a validação.
- O vazio de outro dia pode gerar falso alarme se o treino não tiver um vazio daquele dia.
- Medidas que identificam a sessão (como o formato médio do espectro) dão acurácia falsa de ~100% e foram deixadas de fora. Valide sempre com um **vazio de outro dia** como controle negativo.
- Os CSVs de `dados/` são grandes (alguns passam de 100 MB).
