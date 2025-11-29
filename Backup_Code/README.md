# Traffic System - Mock Demo (Offline Backup)

> **⚠️ EMERGENCY USE ONLY**  
> Use this code **only** if the Wi-Fi fails, the AI server crashes, or you need to demonstrate the hardware without setting up the full Python environment.

## 📖 Overview
This is a **Choreographed Simulation**. It does not use the camera or AI. Instead, it runs a fixed "Movie Script" loop on the ESP32. 

**The Goal:** The system pretends to detect traffic and ambulances. **You (the presenter)** must act out the movement of the cars to match what the lights and gates are doing.

---

## 🎬 The Presentation Script (Choreography)
The code runs in a continuous loop. Memorize this sequence to make the demo look real!

### 🟢 Scene 1: Normal Traffic (Duration: ~15 seconds)
*   **System Action:**
    *   **Road 1** turns <span style="color:green">**GREEN**</span> (5 sec).
    *   Then **Road 2** turns <span style="color:green">**GREEN**</span> (5 sec).
*   **Your Action:**
    *   Move the regular cars across whichever road has the Green light.
    *   Stop cars on the road with the Red light.

### 🚑 Scene 2: Ambulance on Road 1 (Vertical)
*   **The Signal:** 🚧 **Servo 1** (Vertical Road) suddenly moves to **Block the Left Lane**.
*   **System Action:** Road 1 turns/stays <span style="color:green">**GREEN**</span> for 8 seconds.
*   **Your Action:**
    1.  Grab the **Ambulance Toy**.
    2.  Drive it down the **RIGHT LANE** of Road 1.
    3.  *(Explanation to audience: "The system detected an ambulance and closed the left lane to clear a path.")*

### 🔄 Scene 3: Back to Normal
*   **System Action:** Servo 1 opens. Traffic lights cycle normally again.
*   **Your Action:** Move regular cars again.

### 🚑 Scene 4: Ambulance on Road 2 (Horizontal)
*   **The Signal:** 🚧 **Servo 2** (Horizontal Road) suddenly moves to **Block the Right Lane**.
*   **System Action:** Road 2 immediately turns <span style="color:green">**GREEN**</span>.
*   **Your Action:**
    1.  Grab the **Ambulance Toy**.
    2.  Drive it down the **LEFT LANE** of Road 2.
    3.  *(Explanation to audience: "Now it detects an ambulance on the other road and grants priority.")*

---

## ⚙️ Setup & Upload Instructions

1.  **Open Arduino IDE.**
2.  Open the file `Traffic_Mock_Demo.ino`.
3.  **Connect ESP32** to your laptop via USB.
4.  Select your Board: **Tools** > **Board** > **ESP32 Dev Module**.
5.  Select your Port: **Tools** > **Port** > *(Select the COM port)*.
6.  Click **Upload (➡️)**.
7.  **Unplug USB** and power the project via battery (or keep USB plugged in for power).

*Note: You do NOT need to change Wi-Fi settings or IP addresses. This code works completely offline.*

---

## 🔌 Wiring Confirmation
This code uses the exact same wiring as the main AI project. **No hardware changes needed.**

| Component | Pin | Action in Demo |
| :--- | :--- | :--- |
| **Servo 1 (Road 1)** | `GPIO 4` | Blocks Left Lane in Scene 2 |
| **Servo 2 (Road 2)** | `GPIO 2` | Blocks Right Lane in Scene 4 |
| **Road 1 LEDs** | `13, 12, 14` | R, Y, G |
| **Road 2 LEDs** | `27, 26, 25` | R, Y, G |

---

## 🐞 Troubleshooting
*   **Servos Jittering?** Make sure they are powered by 5V (VIN), not 3.3V.
*   **Timing off?** If the servo closes too fast for you to move the car, you can edit the `delay(8000);` line in the code to make it wait longer.