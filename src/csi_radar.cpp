#include "csi_radar_secrets.h"

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"

// UDP proprio deste modulo, separado do "udp" do main.cpp (Wake-on-LAN),
// para nao disputar o mesmo objeto.
static WiFiUDP csiUdp;

static unsigned long ultimo_ping_csi = 0;
// AQUI ESTÁ A ALTERAÇÃO: Reduzido de 150ms para 20ms para atingir ~50 pacotes/s
static const unsigned long intervalo_ping_csi = 20; 

// So aceitamos pacotes CSI com exatamente este tamanho; o resto e descartado.
static const int TAMANHO_CSI = 384;

// Buffer escrito pelo callback (task do WiFi) e lido pelo csiLoop().
// Todo acesso passa pela trava csiMux para nunca ler uma amostra pela metade.
static int8_t bufferCsi[TAMANHO_CSI];
static int8_t rssiAtual = 0;
static unsigned long timestampCaptura = 0;
static volatile bool novoDadoPronto = false;
static portMUX_TYPE csiMux = portMUX_INITIALIZER_UNLOCKED;

// Linha de saida: ate 10 digitos + rssi + tamanho + 384 valores de ate 5 chars + \n
static char linhaBuf[2400];

static void csiCallback(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;
 // if (info->len != TAMANHO_CSI) return; // descarta, nao mascara

  portENTER_CRITICAL(&csiMux);
  memcpy(bufferCsi, info->buf, TAMANHO_CSI);
  rssiAtual = info->rx_ctrl.rssi;
  timestampCaptura = millis();
  novoDadoPronto = true;
  portEXIT_CRITICAL(&csiMux);
}

// Escreve um inteiro em texto no destino e devolve quantos chars escreveu.
static int escreveInt(char *dst, long v) {
  char tmp[12];
  int n = 0;
  bool neg = v < 0;
  unsigned long u = neg ? (unsigned long)(-v) : (unsigned long)v;

  do {
    tmp[n++] = (char)('0' + (u % 10));
    u /= 10;
  } while (u > 0);

  int pos = 0;
  if (neg) dst[pos++] = '-';
  while (n > 0) dst[pos++] = tmp[--n];
  return pos;
}

void csiSetup() {
  // Desliga a economia de energia do WiFi: com modem sleep a recepcao fica irregular.
  WiFi.setSleep(false);

  wifi_csi_config_t csiConfig = {
    .lltf_en = true,
    .htltf_en = true,
    .stbc_htltf2_en = true,
    .ltf_merge_en = true,
    .channel_filter_en = true,
    .manu_scale = false,
    .shift = false,
  };

  if (esp_wifi_set_csi_config(&csiConfig) != ESP_OK) {
    Serial.println("ERRO: esp_wifi_set_csi_config falhou");
  }
  if (esp_wifi_set_csi_rx_cb(csiCallback, NULL) != ESP_OK) {
    Serial.println("ERRO: esp_wifi_set_csi_rx_cb falhou");
  }
  if (esp_wifi_set_csi(true) != ESP_OK) {
    Serial.println("ERRO: esp_wifi_set_csi falhou");
  }

  Serial.println("CSI ativado");
  Serial.println("timestamp_ms,rssi,tamanho,csi_raw");
}

void csiLoop() {
  unsigned long agora = millis();

  if (agora - ultimo_ping_csi >= intervalo_ping_csi) {
    ultimo_ping_csi = agora;
    csiUdp.beginPacket(WiFi.gatewayIP(), 9); // pinga o roteador so pra gerar trafego
    csiUdp.write((uint8_t)0);
    csiUdp.endPacket();
  }

  if (!novoDadoPronto) return;

  // Copia a amostra sob trava e trabalha so na copia.
  int8_t copia[TAMANHO_CSI];
  int8_t rssi;
  unsigned long ts;

  portENTER_CRITICAL(&csiMux);
  memcpy(copia, bufferCsi, TAMANHO_CSI);
  rssi = rssiAtual;
  ts = timestampCaptura;
  novoDadoPronto = false;
  portEXIT_CRITICAL(&csiMux);

  // Monta: timestamp_ms,rssi,tamanho,v0,v1,...,v383\n
  int pos = 0;
  pos += escreveInt(linhaBuf + pos, (long)ts);
  linhaBuf[pos++] = ',';
  pos += escreveInt(linhaBuf + pos, (long)rssi);
  linhaBuf[pos++] = ',';
  pos += escreveInt(linhaBuf + pos, (long)TAMANHO_CSI);

  for (int i = 0; i < TAMANHO_CSI; i++) {
    linhaBuf[pos++] = ',';
    pos += escreveInt(linhaBuf + pos, (long)copia[i]);
  }
  linhaBuf[pos++] = '\n';

  Serial.write((const uint8_t *)linhaBuf, pos);
}