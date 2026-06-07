# Inner Noise — Design Ideas

## Approach A: Biopunk Terminal
**Design Movement:** Biopunk + CRT Terminal Aesthetic  
**Core Principles:** Raw data beauty, organic-meets-digital, monospaced precision, phosphor glow  
**Color Philosophy:** Deep black (#0a0a0a) backgrounds with phosphor green (#00ff88) primary, amber (#ffaa00) warnings, red (#ff3333) danger — evoking a 1980s biological research terminal  
**Layout Paradigm:** Three-panel asymmetric layout: narrow left sidebar (controls), wide center (canvas), right panel (data readouts). No centered hero.  
**Signature Elements:** Scanline overlay on canvas, blinking cursor indicators, monospace data readouts with leading zeros  
**Interaction Philosophy:** Every interaction feels like issuing a command to a living system  
**Animation:** Flicker on state change, phosphor fade-in for new data, pulse on audio activity  
**Typography:** JetBrains Mono for all UI, slightly larger for headings with letter-spacing

<response><text>Biopunk Terminal</text><probability>0.07</probability></response>

---

## Approach B: Brutalist Scientific Instrument
**Design Movement:** New Brutalism + Scientific Instrument Design  
**Core Principles:** Exposed structure, functional honesty, bold contrast, instrument-panel logic  
**Color Philosophy:** Off-white (#f5f0e8) base with deep navy (#0d1b2a) panels, electric cyan (#00d4ff) for active states, warm amber (#e8a020) for warnings — like a vintage oscilloscope  
**Layout Paradigm:** Asymmetric grid with visible structural borders. Left: large canvas. Right: stacked instrument panels. Bottom: timeline/piano roll.  
**Signature Elements:** Thick border rules, exposed grid lines, instrument-style knob/slider aesthetics  
**Interaction Philosophy:** Controls feel physical — sliders have resistance, buttons have weight  
**Animation:** Needle-sweep transitions, oscilloscope-style data drawing, no easing on critical data  
**Typography:** Space Grotesk for headings (bold, condensed), IBM Plex Mono for data values

<response><text>Brutalist Scientific Instrument</text><probability>0.08</probability></response>

---

## Approach C: Void Organism — SELECTED
**Design Movement:** Dark Organic Minimalism + Generative Art Aesthetic  
**Core Principles:** The interface IS the organism; negative space breathes; data has texture; sound has color  
**Color Philosophy:** Near-black (#080c10) void background. Four neuron colors map to the four bottleneck neurons — N1: warm amber (#f5a623), N2: electric teal (#00e5c8), N3: violet (#9b5de5), N4: coral (#ff6b6b). These colors appear consistently across canvas, piano roll, and audio visualizers.  
**Layout Paradigm:** Full-viewport dark canvas as primary surface. Floating instrument panels that slide in from edges. Navigation is a minimal top bar with mode indicators.  
**Signature Elements:** Particle trails behind agent, neuron glow pulses, waveform visualizer strips  
**Interaction Philosophy:** The user is a scientist observing a living system — minimal chrome, maximum signal  
**Animation:** Smooth 60fps canvas, subtle panel fade-ins (200ms ease-out), neuron pulse on activation, glow intensity proportional to activation value  
**Typography:** Space Grotesk (headings, labels), JetBrains Mono (data values, generation counters)

<response><text>Void Organism</text><probability>0.09</probability></response>

---

## Selected: **Approach C — Void Organism**
Deep dark void, four neuron accent colors (amber/teal/violet/coral), Space Grotesk + JetBrains Mono, floating instrument panels, full-viewport canvas.
