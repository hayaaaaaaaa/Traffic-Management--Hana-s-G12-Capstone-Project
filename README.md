# 🚦 AI Smart Traffic Control System (Ambulance Priority)

## 👋 Introduction
This project uses Artificial Intelligence to control traffic lights automatically! 
1.  **The Eye:** Your **Smartphone** acts as the camera.
2.  **The Brain:** Your **Laptop** runs an AI program to count cars and spot Ambulances.
3.  **The Controller:** An **ESP32** turns the LEDs Green/Red and moves the Servo Gates.

**How it works:**
*   If **Road A** has more cars, it gets the Green Light.
*   If an **Ambulance** is seen, that road immediately turns Green, and the Servo Gate blocks the other lane to make space.

---

## 🛠️ Hardware You Need
*   **1x ESP32 Board** (Development Board).
*   **2x Servo Motors** (SG90 or similar) - These act as the "Bar Gates".
*   **6x LEDs** (2 Red, 2 Yellow, 2 Green).
*   **6x Resistors** (220 ohm or 330 ohm).
*   **Jumper Wires & Breadboard**.
*   **1x Smartphone** (Android or iPhone).
*   **1x Laptop/PC**.

---

## 🔌 Wiring Diagram
Connect your components to the ESP32 pins exactly like this:

### 🚦 Traffic Lights
| Road | Color | ESP32 Pin |
| :--- | :--- | :--- |
| **Road 1 (Vertical)** | Red | GPIO 13 |
| | Yellow | GPIO 12 |
| | Green | GPIO 14 |
| **Road 2 (Horizontal)** | Red | GPIO 27 |
| | Yellow | GPIO 26 |
| | Green | GPIO 25 |

### 🚧 Bar Gates (Servo Motors)
| Component | Wire Color | ESP32 Pin |
| :--- | :--- | :--- |
| **Servo 1 (Road 1)** | Orange/Yellow (Signal) | GPIO 4 |
| | Red (Power) | 5V / VIN |
| | Brown/Black (GND) | GND |
| **Servo 2 (Road 2)** | Orange/Yellow (Signal) | GPIO 2 |
| | Red (Power) | 5V / VIN |
| | Brown/Black (GND) | GND |

---

## 💻 Step 1: Install Software on Laptop

### A. Install Python Libraries
1.  Open your computer's **Command Prompt** (Windows) or Terminal (Mac).
2.  Paste this command and hit Enter:
    ```bash
    pip install flask opencv-python numpy pyyaml requests
    ```

### B. Setup Arduino IDE
1.  Install the **Arduino IDE**.
2.  Open it and go to **Tools > Board > Boards Manager**. Search for `ESP32` and install it.
3.  Go to **Sketch > Include Library > Manage Libraries**.
    *   Search for **`ArduinoJson`** -> Click Install.
    *   Search for **`ESP32Servo`** -> Click Install.

---

## 🌐 Step 2: Setup Ngrok (The Secure Tunnel)
*This step allows your phone to connect to your computer securely.*

1.  Go to [ngrok.com/download](https://ngrok.com/download) and download the version for your OS (Windows/Mac).
2.  **Unzip** the downloaded folder. You will see a file named `ngrok.exe`.
3.  Go to [dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup) and create a free account.
4.  On your dashboard, look for **"Your Authtoken"**. Copy that long code.
5.  Open the folder where `ngrok.exe` is.
    *   **Windows Trick:** Click the address bar at the top of the folder window, type `cmd`, and hit Enter. A black box opens.
6.  Paste this command (replace `YOUR_TOKEN` with the code you copied):
    ```bash
    ngrok config add-authtoken YOUR_TOKEN_HERE
    ```
    *It should say "Authtoken saved".*

---

## 🚀 Step 3: Run the System (Do this every time)

**You need 2 Terminal Windows open.**

### Window 1: The AI Brain
1.  Open a terminal in your project folder.
2.  Run the Python server:
    ```bash
    python app.py
    ```
3.  Wait until you see: `Running on http://0.0.0.0:5000`.

### Window 2: The Tunnel
1.  Open the terminal where `ngrok.exe` is located.
2.  Run this command:
    ```bash
    ngrok http 5000
    ```
3.  Look for the line that says **Forwarding**. It will look like this:
    `https://a1b2-c3d4.ngrok-free.app`
4.  **Copy that link.**

### Window 3: The Phone (Camera)
1.  Open Chrome or Safari on your phone.
2.  Paste the **https link** you just copied.
3.  Click **"Allow Camera"**.
4.  You should now see the camera video on your phone!

---

## 🤖 Step 4: Connect the ESP32

1.  On your PC, open Command Prompt and type: `ipconfig`.
2.  Look for **IPv4 Address**. It usually looks like `192.168.1.XX`. Write it down.
3.  Open the file `Traffic_Final.ino` in Arduino IDE.
4.  Find this line at the top:
    ```cpp
    const char* serverUrl = "http://192.168.1.XX:5000/decision";
    ```
5.  **Replace** `192.168.1.XX` with **YOUR** IP address you found in step 1.
    *   *Note: Do NOT use the ngrok link here. Use the number IP.*
6.  Connect your ESP32 to the PC via USB.
7.  Click the **Upload (➡️)** button in Arduino IDE.

---

## ✅ Final Checklist
1.  Is `app.py` running?
2.  Is `ngrok` running?
3.  Is the Phone showing the video?
4.  Is the ESP32 plugged in?

**Testing:**
*   Point the phone at the "Road".
*   If you put a red object (like a toy firetruck) in the frame, the AI should say "Ambulance Detected!" and the servo gate should move!
