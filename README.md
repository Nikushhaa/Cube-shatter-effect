<div align="center">

 
# 🧊 Cube Shatter Effect

<img src="https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif" width="420" />

### 💥 Real-time cube destruction / shatter animation effect

A visually satisfying **3D cube explosion effect** built with smooth motion, fragmentation logic, and clean web animation techniques.

---

<!-- BADGES -->
![GitHub stars](https://img.shields.io/github/stars/Nikushhaa/Cube-shatter-effect?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/Nikushhaa/Cube-shatter-effect?style=for-the-badge)
![License](https://img.shields.io/github/license/Nikushhaa/Cube-shatter-effect?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)

</div>

---

## ✨ Live Concept


🧊 Cube (stable)
↓ click / trigger
💥 explosion starts
↓
🧩 fragments scatter in 3D space
↓
✨ smooth decay / fade-out


---

## ⚡ Features

- 🧊 Realistic cube structure
- 💥 Physics-like shatter explosion effect
- 🎯 Click / event-based trigger system
- 🧩 Fragmented motion system
- 🎨 Clean UI-friendly visual style
- ⚡ Lightweight and fast
- 🧠 Easy to extend into WebGL / Three.js

---

## 🎬 Preview

<div align="center">

<img src="https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif" width="420" />

</div>

---

## 🛠️ Tech Stack

- HTML5
- CSS3 (3D transforms, animations)
- Vanilla JavaScript
- RequestAnimationFrame animation loop
- (Optional upgrade path → Three.js / WebGL)

---

## 🚀 How It Works

### 1. Cube Creation
A cube is rendered using layered HTML/CSS 3D faces.

### 2. Event Trigger
User interaction (click / keypress) activates the effect.

### 3. Shatter Logic
Cube splits into multiple fragments:
- each fragment gets a direction vector
- random rotation is applied
- velocity is assigned

### 4. Animation Loop
Using `requestAnimationFrame`:
- positions update continuously
- fragments move outward
- decay or fade-out applied

---

## 📦 Installation

```bash
git clone https://github.com/Nikushhaa/Cube-shatter-effect.git
cd Cube-shatter-effect
open index.html
📁 Project Structure
Cube-shatter-effect/
│
├── index.html
├── style.css
├── main.js
└── assets/
    ├── textures/
    └── media/
🧩 Usage
Basic setup
<link rel="stylesheet" href="style.css">
<script src="main.js"></script>
Trigger effect
document.addEventListener("click", () => {
  triggerCubeShatter();
});
🎮 Controls
Action	Result
Click	Trigger shatter
Reload	Reset cube
Hover (optional)	Pre-shake effect
🌌 Future Improvements
⚡ WebGL / Three.js upgrade
🎯 Real physics engine integration
🔊 Sound effects on impact
🧠 AI-driven fragmentation patterns
🌈 Shader-based glow effects
📱 Mobile optimization
🧠 Inspiration

Inspired by:

physics-based destruction systems in games
modern UI micro-interactions
WebGL shader animations
cinematic explosion effects
📢 Notes

This project is designed as:

a visual experiment
a frontend animation showcase
a portfolio-level micro-interaction demo
⭐ Support

If you like this project:

⭐ Star the repo
🍴 Fork it
🧠 Improve it
🚀 Share it

<div align="center">
Made with 💙 by Nikushhaa
</div> ```
