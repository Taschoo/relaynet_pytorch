# ReLayNet: Zusammenfassung und Anleitung zum Nachbau

## Einführung

ReLayNet ist ein voll-konvolutionales Netzwerk, das für die End-to-End-Segmentierung von Netzhautschichten und Flüssigkeitsansammlungen in OCT-Bildern entwickelt wurde. Es verwendet Encoder- und Decoder-Blöcke, die durch Skip-Verbindungen verbunden sind, und optimiert die Segmentierung durch eine Kombination von gewichteten Verlustfunktionen.

## Netzwerkarchitektur

### Encoder-Block

-   **Schichten:**
    -   Convolution Layer (Kernelgröße: 7x3)
    -   Batch Normalization Layer
    -   ReLU Activation Layer
    -   Max Pooling Layer
-   **Funktion:** Reduziert die Dimension der Feature-Maps und speichert Pooling-Indizes.

### Decoder-Block

-   **Schichten:**
    -   Unpooling Layer (verwendet gespeicherte Pooling-Indizes)
    -   Concatenation Layer
    -   Convolution Layer (Kernelgröße: 7x3)
    -   Batch Normalization Layer
    -   ReLU Activation Layer
-   **Funktion:** Upsampling der Feature-Maps und Erhalt der räumlichen Konsistenz.

### Klassifikationsblock

-   **Schichten:**
    -   Convolutional Layer (Kernelgröße: 1x1)
    -   Softmax Layer
-   **Funktion:** Reduziert die Kanäle der Feature-Map auf die Anzahl der Klassen und berechnet die Wahrscheinlichkeiten.

## Training und Optimierung

### Verlustfunktionen

-   **Weighted Multi-Class Logistic Loss:** Kompensiert Klassenungleichgewichte.
-   **Dice Loss:** Bewertet die räumliche Überlappung mit dem Ground-Truth.

### Gewichtungsschema

-   **Ziel:** Erhöht die Sensitivität der Kerne für Übergangsbereiche und kompensiert Klassenungleichgewichte.

### Optimierung

-   **Algorithmus:** Stochastische Mini-Batch Gradient Descent mit Momentum und Backpropagation.
-   **Gesamtverlust:** Kombination aus gewichteter logistischer Verlust und Dice Loss.
    ```latex
    J_{overall} = \lambda_1 J_{logloss} + \lambda_2 J_{dice} + \lambda_3 \|W(\cdot)\|_F^2
    ```
