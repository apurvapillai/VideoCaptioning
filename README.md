# VideoCaptioning
using MSVD dataset for video captioning
# At-a-Glance 👀

This project involved designing and implementing a deep learning model capable of automatically generating natural language captions for video content. Using an encoder-decoder architecture based on LSTM networks with an attention mechanism, the system was trained on the MSVD dataset. The encoder extracted frame-level features, while the decoder generated captions word-by-word, guided by dynamic attention weights. The final model achieved a BLEU-4 score of 41.2, a significant improvement of 7.6 points over the baseline. Training time was optimized by employing efficient frame sampling and batch processing techniques.

# Problem

Video content is widely consumed but often lacks accessibility features such as accurate, descriptive captions. This creates barriers for individuals with hearing impairments or those in sound-off environments. Manual captioning is resource-intensive and inconsistent. The challenge was to develop an automated captioning system that understands visual content over time and translates it into coherent and contextually relevant language. Attention mechanisms played a key role in enabling the model to focus on salient frames while generating each word, improving overall semantic quality.

# Solution

To address this, the project employed an encoder-decoder framework based on Long Short-Term Memory (LSTM) networks, enhanced with an attention mechanism. The encoder processed video frames extracted as feature vectors, capturing spatial and temporal information. Instead of encoding all frames equally, the attention mechanism allowed the decoder to dynamically focus on specific frames relevant for generating each word in the caption. This selective focus improved the coherence and relevance of generated captions.

The MSVD dataset, containing thousands of short video clips paired with multiple captions, was used to train and evaluate the model. Feature extraction from videos was optimized by sampling key frames instead of processing every frame, reducing redundancy. Batch processing improvements were introduced to speed up training, balancing memory usage and throughput efficiently.

This approach yielded a BLEU-4 score of 41.2, showing a clear improvement over baseline models that lacked attention or used simpler encoding schemes. The model was able to generate captions that aligned well with human descriptions, demonstrating better understanding of complex visual sequences. The reduction in training time made it more practical for iterative experimentation and deployment.
