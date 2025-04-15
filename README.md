# Detector de Objetos em Tempo Real com TensorFlow

Este projeto é um detector de objetos em tempo real utilizando TensorFlow e OpenCV! A ideia aqui é aproveitar um modelo pré-treinado (SSD MobileNet V2 do TensorFlow Hub) para identificar objetos diretamente pela webcam. Além disso, organizamos o código de forma bacana, separando os rótulos do COCO em um arquivo à parte (`coco_labels.py`), facilitando futuras atualizações e reutilizações.

## Sobre o Projeto

- **Objetivo:**  
  Detectar objetos em tempo real e desenhar caixas delimitadoras, exibindo os nomes dos objetos e suas pontuações de confiança.
  
- **Tecnologias:**  
  - **TensorFlow:** Utilizado para carregar o modelo pré-treinado e realizar a inferência.
  - **OpenCV:** Responsável pela captura do vídeo da webcam e pela exibição dos resultados.
  - **Transfer Learning:** Baseamos nossa solução no SSD MobileNet V2 do TensorFlow Hub (treinado no conjunto de dados COCO), mantendo a implementação simples sem precisar treinar do zero, diferente do projeto anterior.
  


