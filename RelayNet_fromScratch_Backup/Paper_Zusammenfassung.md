# ReLayNet: Retinal Layer and Fluid Segmentation of Macular Optical Coherence Tomography using Fully Convolutional Networks

## Einführung

- **OCT (Optische Kohärenztomographie):** Nicht-invasive Diagnosemethode für diabetisches Makulaödem.
- **ReLayNet:** Voll-konvolutionales Netzwerk für End-to-End-Segmentierung von Netzhautschichten und Flüssigkeitsansammlungen.
- **Architektur:**
  - **Encoder:** Kontrahierende Pfade aus konvolutionalen Blöcken zur Kontextmerkmalserkennung.
  - **Decoder:** Expandierende Pfade aus konvolutionalen Blöcken für semantische Segmentierung.
- **Verlustfunktion:** Kombination aus gewichteter logistischer Regression und Dice-Overlap-Verlust.
- **Validierung:** Vergleich mit fünf aktuellen Segmentierungsmethoden anhand eines öffentlich verfügbaren Benchmark-Datensatzes.

## Stand der Technik

- **Traditionelle Methoden:**
  - Graphenkonstruktionen und dynamische Programmierung zur Schätzung der Schichtgrenzen.
- **Neuere Ansätze:**
  - Kombination von CNNs mit Graph-Suchmethoden zur automatischen Segmentierung der Netzhautschichten.
- **Bildsegmentierungstechniken:**
  - Verwendung von Texturinformationen und Diffusionskarten.
  - Probabilistische Modellierung von Netzhautschichten.
  - Parallele aktive Konturen für gleichzeitige Segmentierung.
- **Voll-konvolutionale Netzwerke (F-CNNs):**
  - Kombination von tiefen, grob aufgelösten Schichten mit flachen, fein aufgelösten Schichten.
  - U-Net-Architektur mit Encoder-Decoder-Framework und Skip-Verbindungen.
  - Effektiv auch bei begrenzten Trainingsdaten durch geeignete Datenaugmentation und Gradienten-Gewichtungsschemata.


## Methodology

### Problem statement

Das Ziel ist es, jedem Pixel eines retinalen OCT-Bildes eine bestimmte Klasse aus einem Labelraum $L = \{ l \} = \{ 1, \cdots, K \}$ für $K$ Klassen zuzuweisen. Diese Klassen umfassen 7 Netzhautschichten, den Bereich über der Retina (RaR), den Bereich unterhalb der RPE (RbR) und angesammelte Flüssigkeiten.

### Network architecture

#### Encoder block

- **Schichten:**
  - Convolution Layer (Kernelgröße: 7x3)
  - Batch Normalization Layer
  - ReLU Activation Layer
  - Max Pooling Layer
  
- **Funktion:** Reduziert die Dimension der Feature-Maps und speichert Pooling-Indizes.

#### Decoder block

- **Schichten:**
  - Unpooling Layer (verwendet gespeicherte Pooling-Indizes)
  - Concatenation Layer
  - Convolution Layer (Kernelgröße: 7x3)
  - Batch Normalization Layer
  - ReLU Activation Layer
  
- **Funktion:** Upsampling der Feature-Maps und Erhalt der räumlichen Konsistenz.

#### Classification block

- **Schichten:**
  - Convolutional Layer (Kernelgröße: 1x1)
  - Softmax Layer
  
- **Funktion:** Reduziert die Kanäle der Feature-Map auf die Anzahl der Klassen und berechnet die Wahrscheinlichkeiten.

### Training

#### Loss functions

- **Weighted Multi-Class Logistic Loss:** Kompensiert Klassenungleichgewichte.
- **Dice Loss:** Bewertet die räumliche Überlappung mit dem Ground-Truth.

#### Weighting scheme for loss function

- **Ziel:** Erhöht die Sensitivität der Kerne für Übergangsbereiche und kompensiert Klassenungleichgewichte.

### Optimization

- **Algorithmus:** Stochastische Mini-Batch Gradient Descent mit Momentum und Backpropagation.
- **Gesamtverlust:** Kombination aus gewichteter logistischer Verlust und Dice Loss.
  $$
  J_{overall} = \lambda_1 J_{logloss} + \lambda_2 J_{dice} + \lambda_3 \|W(\cdot)\|_F^2
  $$

### OCT B-scan slicing and data augmentation

- **Slicing:** Unterteilung der OCT-B-Scans in kleinere Linien.
- **Datenaugmentation:** Zufällige horizontale Spiegelungen und räumliche Verschiebungen.

## Experimentelles Setup

### Datensatz
- **Duke SD-OCT Dataset:** Öffentlich verfügbarer Datensatz für Patienten mit diabetischem Makulaödem (DME).
- **Anzahl der B-Scans:** 110 annotierte B-Scans von 10 Patienten (jeweils 11 B-Scans pro Patient).
- **Annotationen:** Zwei Experten annotierten die Netzhautschichten und Flüssigkeitsregionen.

### Experimentelle Einstellungen
- **Datenaufteilung:** 
  - Training: Subjekte 1-5 (55 B-Scans)
  - Test: Subjekte 6-10 (55 B-Scans)
- **Hyperparameter:**
  - $\lambda_1 = 1$, $\lambda_2 = 0.5$, $\lambda_3 = 0.0001$
  - $\omega_1 = 10$, $\omega_2 = 5$
- **Optimierung:**
  - Mini-Batches: Größe = 50
  - Lernrate: Startet bei 0.1, Reduktion um eine Größenordnung nach jeweils 30 Epochen
  - Momentum: 0.9
- **Hardware:** Intel Xeon CPU, 12 GB Nvidia Tesla K40 GPU, 64 GB RAM

### Vergleichsmethoden und Baselines
- **State-of-the-Art Methoden:**
  - **CM-GDP:** Graph-basierte dynamische Programmierung
  - **CM-KR:** Kernel-Regression mit GDP
  - **CM-LSE:** Layer-spezifische strukturierte Kantenerkennung mit GDP
  - **CM-Unet:** U-Net Architektur
  - **CM-FCN:** Voll-konvolutionales Netzwerk
- **Baselines:**
  - Verschiedene Architekturen und Verlustfunktion-Kombinationen, um die Wichtigkeit der einzelnen Komponenten von ReLayNet zu bewerten.

### Bewertungsmetriken
- **Dice Overlap Score (DS):** Misst die Übereinstimmung zwischen der vorhergesagten Segmentierung und dem Ground-Truth.
- **Contour Error (CE):** Schätzfehler für jede Schichtgrenze.
- **Mean Absolute Difference in Layer Thickness (MAD-LT):** Fehler in der geschätzten Schichtdicke.

## Experimentelle Beobachtungen und Diskussion

### Qualitativer Vergleich von ReLayNet mit Vergleichsmethoden
- **Fall 1:** Pathologischer OCT B-Scan mit DME:
  - ReLayNet segmentiert erfolgreich kleine Flüssigkeitspools, wo andere Methoden versagen.
  - ReLayNet und CM-Unet zeigen hochwertige Segmentierungen vergleichbar mit Expertenannotationen.
- **Fall 2:** OCT B-Scan ohne Flüssigkeitsansammlung:
  - Konsistente Leistung aller Vergleichsmethoden.
  - ReLayNet zeigt verbesserte Details und Genauigkeit in der Schichtsegmentierung.

### Quantitativer Vergleich von ReLayNet mit Vergleichsmethoden
- **Metriken:**
  - ReLayNet zeigt die beste Segmentierungseffizienz in 9 von 10 Klassen.
  - Höchste Dice Scores für mehrere Schichten und Flüssigkeitsklassen.
  - Übertrifft andere Methoden bei der Schichtdickenabschätzung und Konturgenauigkeit.

### Bedeutung der Beiträge von ReLayNet
- **Skip-Verbindungen:**
  - Verbesserung der Segmentierungsleistung durch Hinzufügen von Kontextinformationen.
  - Wesentliche Verbesserung der Dice Scores und Reduktion der Fehler in der Schichtdickenabschätzung und Konturgenauigkeit.
- **Verlustfunktionen:**
  - Kombination von gewichteter logistischer Verlust und Dice Loss führt zu besseren Ergebnissen.
  - Verbesserung der Leistung bei kritischen Schichten durch kombinierte Verluste.
- **Netzwerktiefe:**
  - ReLayNet (3-1-3 Konfiguration) bietet eine optimale Balance zwischen Modellkomplexität und Leistung.
- **Gewichtetes Verlustschema:**
  - Verbessert die Segmentierung von Schichtübergängen und kompensiert Klassenungleichgewichte.

### Gefaltete Kreuzvalidierung
- **8-fach Kreuzvalidierung:**
  - Signifikante Leistungssteigerung durch Ensemble-Ansatz.
  - Verbesserte Dice Scores und konsistente Ergebnisse über verschiedene Falten.

### Analyse des ETDRS-Rasters über Patienten
- **ETDRS-Raster:**
  - Berechnung der durchschnittlichen Netzhautdicke in 9 Zonen.
  - ReLayNet zeigt die geringsten Fehler in der Schichtdickenabschätzung für alle Zonen.
  - Verbesserte Schätzgenauigkeit insbesondere in der fovealen Region.

## Fazit
- **ReLayNet:** Überlegene Segmentierung von Netzhautschichten und Flüssigkeitsansammlungen in OCT-Bildern.
- **Hauptbeiträge:**
  - End-to-End voll-konvolutionales Netzwerk
  - Kombination von gewichteter logistischer Verlust und Dice Loss
  - Verwendung von Skip-Verbindungen und Unpooling-Layern
- **Leistung:** ReLayNet zeigt verbesserte Segmentierungsgenauigkeit und -konsistenz, auch bei pathologischen Variationen.
- **Zukunft:** Erweiterung von ReLayNet auf intraoperative Szenarien und 3D-Volumensegmentierung mit größerem Trainingsdatensatz.
