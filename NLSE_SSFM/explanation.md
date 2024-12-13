This MATLAB code simulates the propagation of optical pulses through a fiber optic system based on the nonlinear Schrödinger equation. The simulation includes both symmetric and asymmetric split-step Fourier methods. Here's an explanation of the key components:

---

### **Constants and Initial Setup**
- **`p0`**, **`gamma`**, **`beta2`**, and **`l`**: Define the initial parameters such as input power, nonlinear coefficient, dispersion coefficient, and fiber length.
- **`t0`**: Temporal width derived from the soliton condition.
- **`t`**: Time axis for simulating the pulse.
- **`Ts`** and **`fs`**: Sampling interval and frequency.
- **`dz`**: Step size for the propagation distance.

---

### **Input Pulses**
1. **Part 1: Soliton Pulse**
   - A hyperbolic secant pulse is used as the input.

2. **Part 2: Gaussian Pulse**
   - A Gaussian pulse with an optional chirp parameter \( C \) is used.

---

### **Operators**
- **`D_operator`**: Applies dispersion effects in the frequency domain using the dispersion parameter \( \beta_2 \).
- **`N_operator`**: Applies nonlinear effects in the time domain using the nonlinearity coefficient \( \gamma \).

---

### **Split-Step Fourier Method**
1. **Symmetric Scheme**:
   - Splits each step symmetrically: \( D(z/2) \to N(z) \to D(z/2) \).
   - Ensures higher accuracy in handling nonlinear and dispersive effects.
   
2. **Asymmetric Scheme**:
   - Uses a simpler \( D(z) \to N(z) \) sequence.
   - Computationally less accurate than the symmetric approach.

---

### **Visualization**
- **Subplots**: Two subplots for each pulse type (Soliton and Gaussian) compare:
  - Input pulse vs. Output pulse (Symmetric).
  - Input pulse vs. Output pulse (Asymmetric).

- **Plot Details**:
  - Time axis in nanoseconds.
  - Colors and styles distinguish between input and output pulses.

---

### **Key Features**
- **FFT/IFFT**: Fast Fourier Transform and its inverse are used for switching between the time and frequency domains.
- **Grid Resolution**: Fine temporal resolution ensures accurate spectral representation.
- **Legend and Titles**: Clearly label the input and output pulses for each case.

---

This code is a robust framework for optical pulse propagation studies. If you want to enhance or customize the simulation further, let me know!
