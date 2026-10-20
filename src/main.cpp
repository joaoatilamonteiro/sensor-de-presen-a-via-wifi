#include <Arduino.h>
#include <WiFi.h>
#include "csi_radar_secrets.h"

// Declaração das funções que estão no ficheiro csi_radar.cpp
extern void csiSetup();
extern void csiLoop();

void setup() {
  // Inicializa a serial com a taxa alta para evitar truncamento dos 384 valores
  Serial.begin(921600);
  
  // Aguarda a porta serial estabilizar
  while (!Serial) {
    delay(10);
  }

  Serial.println("\nIniciando WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi conectado!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  // Inicializa a configuração de captura de CSI
  csiSetup();
}

void loop() {
  // Mantém a captura e o envio dos dados rodando
  csiLoop();
}