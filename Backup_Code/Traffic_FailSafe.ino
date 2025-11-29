/*
 * TRAFFIC SYSTEM - MOCK DEMO (CHOREOGRAPHED)
 * 
 * USE CASE:
 * This acts out a script to demonstrate hardware features (Servos + Lights)
 * without requiring the AI or Wi-Fi to actually work.
 * 
 * SCRIPT LOOP:
 * 1. Normal Traffic Cycle (Road 1 -> Road 2)
 * 2. EMERGENCY SIMULATION: Ambulance on Road 1 (Servo 1 closes)
 * 3. Normal Traffic Cycle
 * 4. EMERGENCY SIMULATION: Ambulance on Road 2 (Servo 2 closes)
 */

#include <ESP32Servo.h> 

// --- PINS ---
#define R1_RED 13
#define R1_YELLOW 12
#define R1_GREEN 14

#define R2_RED 27
#define R2_YELLOW 26
#define R2_GREEN 25

#define SERVO1_PIN 4  // Road 1 Gate
#define SERVO2_PIN 2  // Road 2 Gate

Servo servo1;
Servo servo2;

// Standard Angles
int OPEN_GATE = 90;
int CLOSE_LEFT = 0;
int CLOSE_RIGHT = 180;

void setup() {
  Serial.begin(115200);
  
  // Setup LEDs
  pinMode(R1_RED, OUTPUT); pinMode(R1_YELLOW, OUTPUT); pinMode(R1_GREEN, OUTPUT);
  pinMode(R2_RED, OUTPUT); pinMode(R2_YELLOW, OUTPUT); pinMode(R2_GREEN, OUTPUT);

  // Setup Servos
  servo1.setPeriodHertz(50); 
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.setPeriodHertz(50); 
  servo2.attach(SERVO2_PIN, 500, 2400);

  // Initialize: All Red, Gates Open
  servo1.write(OPEN_GATE);
  servo2.write(OPEN_GATE);
  digitalWrite(R1_RED, HIGH); digitalWrite(R2_RED, HIGH);
  
  delay(2000); // Wait 2 seconds before starting
}

void loop() {
  // ==========================================
  // SCENE 1: NORMAL TRAFFIC FLOW
  // ==========================================
  Serial.println("--- NORMAL TRAFFIC START ---");
  
  // Road 1 Green (5 Seconds)
  setLights("ROAD1");
  delay(5000);
  
  // Road 1 Yellow -> Red
  setLights("YELLOW_TO_R2");
  delay(2000);
  
  // Road 2 Green (5 Seconds)
  setLights("ROAD2");
  delay(5000);

  // Road 2 Yellow -> Red
  setLights("YELLOW_TO_R1");
  delay(2000);

  // ==========================================
  // SCENE 2: AMBULANCE ON ROAD 1 (VERTICAL)
  // ==========================================
  Serial.println("!!! AMBULANCE DETECTED ON ROAD 1 !!!");
  
  // 1. Force Road 1 Green immediately
  setLights("ROAD1");
  
  // 2. SIMULATION: Ambulance is in Right Lane, so we block Left Lane (Servo 0)
  // "Closing Gate to protect Ambulance Path..."
  servo1.write(CLOSE_LEFT); 
  
  // 3. Hold this state for 8 seconds (Time for friend to move toy ambulance)
  delay(8000);
  
  // 4. Reset Servo
  servo1.write(OPEN_GATE);
  Serial.println("--- Ambulance Passed ---");
  
  // Transition back to normal
  setLights("YELLOW_TO_R2");
  delay(2000);

  // ==========================================
  // SCENE 3: NORMAL TRAFFIC BUFFER
  // ==========================================
  // Let Road 2 have a turn normally
  setLights("ROAD2");
  delay(5000);
  setLights("YELLOW_TO_R1");
  delay(2000);

  // ==========================================
  // SCENE 4: AMBULANCE ON ROAD 2 (HORIZONTAL)
  // ==========================================
  Serial.println("!!! AMBULANCE DETECTED ON ROAD 2 !!!");
  
  // 1. Force Road 2 Green
  setLights("ROAD2");
  
  // 2. SIMULATION: Ambulance in Left Lane, so we block Right Lane (Servo 180)
  servo2.write(CLOSE_RIGHT);
  
  // 3. Hold for 8 seconds
  delay(8000);
  
  // 4. Reset Servo
  servo2.write(OPEN_GATE);
  
  // Transition back
  setLights("YELLOW_TO_R1");
  delay(2000);
  
  // Loop restarts...
}

// Helper function to handle lights easily
void setLights(String state) {
  // Turn off all Yellows first
  digitalWrite(R1_YELLOW, LOW); digitalWrite(R2_YELLOW, LOW);

  if (state == "ROAD1") {
    digitalWrite(R1_GREEN, HIGH); digitalWrite(R1_RED, LOW);
    digitalWrite(R2_GREEN, LOW); digitalWrite(R2_RED, HIGH);
  }
  else if (state == "ROAD2") {
    digitalWrite(R1_GREEN, LOW); digitalWrite(R1_RED, HIGH);
    digitalWrite(R2_GREEN, HIGH); digitalWrite(R2_RED, LOW);
  }
  else if (state == "YELLOW_TO_R2") {
    digitalWrite(R1_GREEN, LOW); digitalWrite(R1_RED, LOW); digitalWrite(R1_YELLOW, HIGH);
    digitalWrite(R2_RED, HIGH);
  }
  else if (state == "YELLOW_TO_R1") {
    digitalWrite(R2_GREEN, LOW); digitalWrite(R2_RED, LOW); digitalWrite(R2_YELLOW, HIGH);
    digitalWrite(R1_RED, HIGH);
  }
}
