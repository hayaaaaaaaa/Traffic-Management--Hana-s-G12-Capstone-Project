#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h> 

const char* ssid = "HHJM";
const char* password = "seuzcanal75";
const char* serverUrl = "http://192.168.1.18:5000/decision"; // Update IP

// --- Traffic Light Pins ---
// Road 1 (Vertical)
#define R1_RED 13
#define R1_YELLOW 12
#define R1_GREEN 14

// Road 2 (Horizontal)
#define R2_RED 27
#define R2_YELLOW 26
#define R2_GREEN 25

// --- Servo Pins ---
#define SERVO1_PIN 4  // Road 1 Gate
#define SERVO2_PIN 2  // Road 2 Gate

Servo servo1;
Servo servo2;

// Variables
String currentGreen = "ROAD1";
int targetServo1 = 90;
int targetServo2 = 90;
unsigned long lastCheck = 0;

void setup() {
  Serial.begin(115200);
  
  // Lights
  pinMode(R1_RED, OUTPUT); pinMode(R1_YELLOW, OUTPUT); pinMode(R1_GREEN, OUTPUT);
  pinMode(R2_RED, OUTPUT); pinMode(R2_YELLOW, OUTPUT); pinMode(R2_GREEN, OUTPUT);

  // Servos
  servo1.setPeriodHertz(50); 
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.setPeriodHertz(50); 
  servo2.attach(SERVO2_PIN, 500, 2400);
  
  // Initialize Servos to Open (Up)
  servo1.write(90);
  servo2.write(90);

  WiFi.begin(ssid, password);
  while(WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi Connected!");
}

void loop() {
  // 1. Get Data from Cloud (every 500ms)
  if (millis() - lastCheck > 500) {
    if(WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(serverUrl);
      int httpCode = http.GET();
      
      if(httpCode > 0) {
        String payload = http.getString();
        StaticJsonDocument<200> doc;
        deserializeJson(doc, payload);
        
        // Parse JSON
        const char* light = doc["light"];
        currentGreen = String(light);
        targetServo1 = doc["servo1"];
        targetServo2 = doc["servo2"];
        
        Serial.printf("Target: %s | S1: %d | S2: %d\n", currentGreen.c_str(), targetServo1, targetServo2);
      }
      http.end();
    }
    lastCheck = millis();
  }

  // 2. Actuate Hardware
  controlLights();
  controlServos();
}

void controlServos() {
  // Smoothly move servos or just write directly
  servo1.write(targetServo1);
  servo2.write(targetServo2);
}

void controlLights() {
  // Reset Yellows
  digitalWrite(R1_YELLOW, LOW); digitalWrite(R2_YELLOW, LOW);

  if (currentGreen == "ROAD1") {
    digitalWrite(R1_GREEN, HIGH); digitalWrite(R1_RED, LOW);
    digitalWrite(R2_GREEN, LOW); digitalWrite(R2_RED, HIGH);
  } else {
    digitalWrite(R1_GREEN, LOW); digitalWrite(R1_RED, HIGH);
    digitalWrite(R2_GREEN, HIGH); digitalWrite(R2_RED, LOW);
  }
}
